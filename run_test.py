from dataset_transforms import PositionalRandomReplaceVector, ClinicalDataset
import torch
from torch.utils.data import DataLoader
from evaluate import evaluate
def run_test(model, feats_to_keep, test_df, target_idx, target_cols):
    result_acc= [0] * len(feats_to_keep)
    result_ba = [0] * len(feats_to_keep)
    result_auc= [0] * len(feats_to_keep)
    for idx in range(len(feats_to_keep)):
        p_vec = torch.ones(len(test_df.columns), dtype=torch.float32)
        
        # Set features to keep → 0
        mask_keep = torch.tensor(test_df.columns.isin(feats_to_keep[idx]), dtype=torch.bool)
        p_vec[mask_keep] = 0
        
        # Set target cols → 1, but only if they are NOT in feats_to_keep
        mask_target = torch.tensor(test_df.columns.isin(target_cols), dtype=torch.bool)
        mask_target = mask_target & ~mask_keep   # remove overlap
        p_vec[mask_target] = 1
        
        print(p_vec)
        drop_transform_test=PositionalRandomReplaceVector(p_vec, value=16)
        test_dataset = ClinicalDataset(test_df, drop_transform=drop_transform_test)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        test_loss,test_acc,test_ba, auc_test,cm = evaluate(model,test_loader,target_idx)
        print(f"Test Acc: {test_acc:.4f} |"
              f"Test Auc: {auc_test:.4f} ")
        print('Test cm',cm[0:2,0:2])
        result_acc[idx] = test_acc
        result_ba[idx] = test_ba
        result_auc[idx] = auc_test
    return result_acc, result_ba, result_auc