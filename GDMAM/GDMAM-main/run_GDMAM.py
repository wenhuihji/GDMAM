import time
from methods.GDMAM import GDMAM
from utils.resample_loss import ResampleLoss
from utils.utils import set_seed
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from utils.metrics import evaluation_KLD
import argparse
from utils.metrics import evaluation_lt
import copy
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sample_data')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--lambda1', type=float, default=1)
parser.add_argument('--lambda2', type=float, default=1)
parser.add_argument('--lambda3', type=float, default=1)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--valid_size', type=int, default=20) 
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_folds', type=int, default=10)    

# -----------------------------
# 组网函数
# -----------------------------
def get_model():
    global x_train, y_train, hidden_dim, lambda1, lambda2, lambda3, lr, max_epoch, device

    model = GDMAM(
        loss_func=ResampleLoss(
            reweight_func='rebalance', loss_weight=1.0,
            focal=dict(focal=True, alpha=0.5, gamma=2),
            logit_reg=dict(init_bias=0.05, neg_scale=2.0),
            map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
            class_freq=np.sum(y_train, axis=0), train_num=x_train.shape[0],
            reduction='mean'
        ),
        num_feature=x_train.shape[1],
        num_classes=y_train.shape[1],
        hidden_dim=hidden_dim,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        lr=lr,
        weight_decay=1e-4,
        adjust_lr=False,
        gradient_clip_value=5.0,
        max_epoch=max_epoch,
        verbose=False,
        device=device
    )
    return model


def _train():
    global train_loader, val_loader, train_path, device, max_epoch
    print('Start Training!')
    best_state_dict = None
    model = get_model()
    model.to(device)
    min_val_kld = np.inf

    train_kld_curve = []
    val_kld_curve = []

    for epoch in range(max_epoch):
        model.train_loop(epoch=epoch, train_loader=train_loader)

        preds_train, ys_train = model.get_result(test_loader=train_loader)
        train_kld = evaluation_KLD(ys_train, preds_train)

        preds_val, ys_val = model.get_result(test_loader=val_loader)
        val_kld = evaluation_KLD(ys_val, preds_val)

        train_kld_curve.append(train_kld)
        val_kld_curve.append(val_kld)

        msg = f'Epoch {epoch + 1}/{max_epoch} - Train KLD: {train_kld:.6f} | Val KLD: {val_kld:.6f}'
        if val_kld < min_val_kld:
            min_val_kld = val_kld
            best_state_dict = copy.deepcopy(model.state_dict())
            msg += ' * (Best Model Updated)'

    os.makedirs(train_path, exist_ok=True)
    torch.save({'model': best_state_dict}, os.path.join(train_path, 'best.tar'))
    model.save(train_path, epoch=max_epoch - 1)

    return np.array(train_kld_curve), np.array(val_kld_curve), float(min_val_kld)

def _test():
    global test_loader, train_path, device
    print('Start Testing!')
    model = get_model()
    model.to(device)

    checkpoint = torch.load(os.path.join(train_path, 'best.tar'), map_location=device)
    model.load_state_dict(checkpoint['model'])

    preds, ys = model.get_result(test_loader=test_loader)
    return preds


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset

    hidden_dim = args.hidden_dim
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    lambda3 = args.lambda3
    lr = args.lr
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    valid_size = args.valid_size
    device = args.device
    seed = args.seed
    n_folds = args.n_folds
    set_seed(seed)

    method = 'GDMAM'

    mat_file_path = os.path.join(r'E:\实验\GDMAM\dataset', f'{dataset}.mat')
    if not os.path.isfile(mat_file_path):
        mat_file_path = os.path.join('data', f'{dataset}.mat')
        if not os.path.isfile(mat_file_path):
            raise FileNotFoundError(f"数据文件不存在: {mat_file_path}")

    print(f"Loading data from: {mat_file_path}")
    mat_data = scipy.io.loadmat(mat_file_path)

    feat_keys = ['features', 'X', 'data', 'feature', 'Inputs']
    label_keys = ['labels', 'Y', 'label', 'Targets']

    def _find_key(d, candidates):
        for k in candidates:
            if k in d:
                return k
        raise KeyError(f"在 .mat 中未找到任一候选键：{candidates}")

    X_key = _find_key(mat_data, feat_keys)
    Y_key = _find_key(mat_data, label_keys)

    X_full = mat_data[X_key].astype(np.float32)
    Y_full = mat_data[Y_key].astype(np.float32)
    if Y_full.ndim == 1:
        Y_full = Y_full[:, None]

    if device.lower().startswith('cuda') and not torch.cuda.is_available():
        print("[Warn] CUDA 不可用，自动切换到 CPU。")
        device = 'cpu'

    print(f'dataset: {dataset}, hidden_dim: {hidden_dim}, lambda1:{lambda1}, lambda2: {lambda2}, lambda3: {lambda3}')
    base_train_root = os.path.join('save', 'lt', method, 'train')
    base_result_root = os.path.join('save', 'lt', method)
    os.makedirs(base_train_root, exist_ok=True)
    os.makedirs(base_result_root, exist_ok=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    metric_names = ['Chebyshev', 'Clark', 'Canberra', 'KLD', 'Cosine', 'Intersection']
    metrics_all = {name: [] for name in metric_names}

    for fold_idx, (train_index_all, test_index) in enumerate(kf.split(X_full), start=1):
        print("\n" + "=" * 70)
        print(f"[Fold {fold_idx}/{n_folds}]")

        current_x_train_all = X_full[train_index_all]
        current_y_train_all = Y_full[train_index_all]
        x_test = X_full[test_index]
        y_test = Y_full[test_index]

        y_train_all_eval = current_y_train_all

        effective_valid_size = min(valid_size, max(1, current_x_train_all.shape[0] - 1))
        train_index, val_index = train_test_split(
            np.arange(current_x_train_all.shape[0]),
            test_size=effective_valid_size,
            shuffle=True,
            random_state=seed + fold_idx  
        )

 
        x_val, y_val = current_x_train_all[val_index], current_y_train_all[val_index]
        x_train, y_train = current_x_train_all[train_index], current_y_train_all[train_index]

        train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

       
        train_path = os.path.join(base_train_root, f'{dataset}_fold{fold_idx}')
        result_path = base_result_root  

        result_npz_path = os.path.join(result_path, f'{dataset}_fold{fold_idx}.npz')
        if os.path.exists(result_npz_path):
            print(method, dataset, f'Fold {fold_idx} exists! Load results.')
            fold_result = np.load(result_npz_path, allow_pickle=True)
            y_pred = fold_result['y_pred']
          
        else:
            print(method, dataset, f'Fold {fold_idx} training!')
            set_seed(seed + fold_idx)  

            t = time.time()
            train_kld_curve, val_kld_curve, best_val_kld = _train()
            training_time = time.time() - t


            t = time.time()
            y_pred = _test()
            test_time = time.time() - t


            np.savez(
                result_npz_path,
                y_pred=y_pred,
                training_time=training_time,
                test_time=test_time,
                train_kld_curve=train_kld_curve,
                val_kld_curve=val_kld_curve,
                best_val_kld=best_val_kld
            )


        fold_metrics = evaluation_lt(y_test, y_pred, y_train=y_train_all_eval)

        Chebyshev = float(fold_metrics['Chebyshev'])
        Clark = float(fold_metrics['Clark'])
        Canberra = float(fold_metrics['Canberra'])
        KLD = float(fold_metrics['KLD'])
        Cosine = float(fold_metrics['Cosine'])
        Intersection = float(fold_metrics['Intersection'])

        print(f'[Fold {fold_idx}] Chebyshev:   {Chebyshev:.4f}')
        print(f'[Fold {fold_idx}] Clark:       {Clark:.4f}')
        print(f'[Fold {fold_idx}] Canberra:    {Canberra:.4f}')
        print(f'[Fold {fold_idx}] KLD:         {KLD:.4f}')
        print(f'[Fold {fold_idx}] Cosine:      {Cosine:.4f}')
        print(f'[Fold {fold_idx}] Intersection:{Intersection:.4f}')

        # 收集指标
        metrics_all['Chebyshev'].append(Chebyshev)
        metrics_all['Clark'].append(Clark)
        metrics_all['Canberra'].append(Canberra)
        metrics_all['KLD'].append(KLD)
        metrics_all['Cosine'].append(Cosine)
        metrics_all['Intersection'].append(Intersection)

    print("\n" + "=" * 70)
    print(f"[{dataset}] {n_folds} 折交叉验证汇总（均值 ± 最大绝对偏差）")
    summary = {}
    for name in metric_names:
        values = np.array(metrics_all[name], dtype=np.float64)
        mean_v = float(np.mean(values))
        max_abs_dev = float(np.max(np.abs(values - mean_v)))
        summary[name] = dict(mean=mean_v, max_abs_dev=max_abs_dev, per_fold=values.tolist())
        print(f'{name}: {mean_v:.4f} ± {max_abs_dev:.4f}')

    # 可选：保存汇总 npz
    summary_npz = os.path.join(base_result_root, f'{dataset}_cv_summary.npz')
    np.savez(
        summary_npz,
        metrics_all={k: np.array(v, dtype=np.float64) for k, v in metrics_all.items()},
        summary=summary,
        n_folds=n_folds
    )
    print(f"汇总结果已保存到: {summary_npz}")
