from typing import List

import pandas as pd
from pytorch_lightning import Callback

class HNMCallback(Callback):
    def __init__(self,
                 base_df: pd.DataFrame,
                 train_idx: List[int],
                 cfg):
        self.base_df = base_df
        self.aug_df = base_df
        self.cfg       = cfg
        self.aug_cnt   = {c: 0 for c in base_df["target"].unique()}

        self.is_running = self.cfg.trainer.hnm.use_hnm

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.cfg.trainer.hnm.stop_epoch:
            if self.is_running:
                print('HNM Callback Stopped')
                self.is_running = False
        
        if not self.is_running:
            return
        per_cls = pl_module.per_class_loss()
        epoch   = trainer.current_epoch

        max_aug = 5
        topk    = 3

        worst = [cls for cls, _ in sorted(per_cls.items(),
                                          key=lambda kv: kv[1],
                                          reverse=True)
                 if self.aug_cnt[cls] < max_aug][:topk]
        if not worst:
            return

        # # fold-train subset에서 해당 클래스 행 복제
        # subset = self.base_df.iloc[self.train_idx]
        # aug_df = subset[subset["target"].isin(worst)].copy()
        # new_df = pd.concat([self.base_df, aug_df], ignore_index=True)

        new_aug_df = self.base_df[self.base_df["target"].isin(worst)].copy()

        self.aug_df = pd.concat([self.aug_df, new_aug_df], ignore_index=True)

        # 카운트 증가
        for cls in worst:
            self.aug_cnt[cls] += 1

        # train_tf = trainer.datamodule.train_ds.transform
        trainer.datamodule.set_train_dataset(self.aug_df)

        print(f"[HNM] Epoch {epoch+1} → Worst Class {worst} Copy(+{len(new_aug_df)}) "
              f"Total {len(self.aug_df)}")
