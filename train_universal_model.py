"""
UniversalFragranceGenerator 최적화 학습 스크립트
===================================================

이 스크립트는 UniversalFragranceGenerator 모델을 위한 최적화된 학습 파이프라인을 제공합니다.

주요 기능:
- AdamW 옵티마이저 + Cosine Annealing with Warm Restarts
- Mixed Precision Training (AMP)
- Gradient Accumulation (메모리 절약)
- Label Smoothing
- Early Stopping
- Best Model Checkpoint
- TensorBoard 로깅
- Learning Rate Finder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from sqlalchemy.orm import Session
from tqdm import tqdm
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

from fragrance_ai.models.deep_learning_architecture import (
    UniversalFragranceGenerator,
    FragranceGenerationConfig
)
from fragrance_ai.database.connection import get_session
from fragrance_ai.database.models import Recipe, RecipeIngredient, FragranceNote

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """학습 설정"""

    # 모델 설정
    model_config_path: Optional[str] = None

    # 데이터 설정
    batch_size: int = 16
    gradient_accumulation_steps: int = 4  # 실제 배치 크기 = batch_size * gradient_accumulation_steps
    num_workers: int = 4
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    max_sequence_length: int = 50

    # 학습 설정
    max_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # 스케줄러 설정
    T_0: int = 10  # Cosine Annealing 주기
    T_mult: int = 2  # 주기 증가 배수
    eta_min: float = 1e-6  # 최소 학습률

    # 정규화
    label_smoothing: float = 0.1
    dropout: float = 0.1

    # Early Stopping
    patience: int = 10
    min_delta: float = 1e-4

    # 체크포인트
    checkpoint_dir: str = "checkpoints/universal_fragrance"
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3

    # Mixed Precision Training
    use_amp: bool = True

    # TensorBoard
    log_dir: str = "runs/universal_fragrance"
    log_every_n_steps: int = 10

    # 기타
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    resume_from_checkpoint: Optional[str] = None


class FragranceDataset(Dataset):
    """향수 레시피 데이터셋"""

    def __init__(self,
                 recipes: List[Recipe],
                 tokenizer: AutoTokenizer,
                 note_to_idx: Dict[str, int],
                 max_length: int = 50):
        """
        Args:
            recipes: 레시피 리스트
            tokenizer: 텍스트 토크나이저
            note_to_idx: 향료 노트 -> 인덱스 매핑
            max_length: 최대 시퀀스 길이
        """
        self.recipes = recipes
        self.tokenizer = tokenizer
        self.note_to_idx = note_to_idx
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        recipe = self.recipes[idx]

        # 텍스트 인코딩 (레시피 설명)
        text = f"{recipe.name}. {recipe.description or ''} {recipe.concept or ''}"
        text_encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 재료 정보 추출
        ingredients = sorted(recipe.ingredients, key=lambda x: x.note_position)

        # 시퀀스 생성
        notes = []
        concentrations = []
        volumes = []

        for ingredient in ingredients[:self.max_length]:
            # 노트 인덱스
            note_idx = self.note_to_idx.get(
                ingredient.note.name,
                self.note_to_idx.get('[UNK]', 0)
            )
            notes.append(note_idx)

            # 농도 (0-9 레벨로 이산화)
            concentration_level = min(int(ingredient.percentage / 10), 9)
            concentrations.append(concentration_level)

            # 부피 (정규화된 값)
            volume = ingredient.percentage / 100.0
            volumes.append(volume)

        # 패딩
        seq_len = len(notes)
        if seq_len < self.max_length:
            padding_length = self.max_length - seq_len
            notes.extend([0] * padding_length)  # PAD token
            concentrations.extend([0] * padding_length)
            volumes.extend([0.0] * padding_length)

        return {
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
            'target_notes': torch.tensor(notes, dtype=torch.long),
            'target_concentrations': torch.tensor(concentrations, dtype=torch.long),
            'target_volumes': torch.tensor(volumes, dtype=torch.float),
            'sequence_length': torch.tensor(seq_len, dtype=torch.long)
        }


class LabelSmoothingCrossEntropy(nn.Module):
    """Label Smoothing Cross Entropy Loss"""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, L) - 예측 로짓
            target: (B, L) - 타겟 인덱스
        """
        n_classes = pred.size(1)

        # One-hot encoding with label smoothing
        one_hot = torch.zeros_like(pred).scatter_(1, target.unsqueeze(1), 1)
        smooth_one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes

        # Log softmax
        log_probs = F.log_softmax(pred, dim=1)

        # Cross entropy
        loss = -(smooth_one_hot * log_probs).sum(dim=1).mean()

        return loss


class EarlyStopping:
    """Early Stopping"""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class Trainer:
    """학습 파이프라인"""

    def __init__(self,
                 model: UniversalFragranceGenerator,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # 디바이스 설정
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 옵티마이저
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 스케줄러
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.eta_min
        )

        # Loss functions
        self.note_loss_fn = LabelSmoothingCrossEntropy(config.label_smoothing)
        self.concentration_loss_fn = LabelSmoothingCrossEntropy(config.label_smoothing)
        self.volume_loss_fn = nn.MSELoss()

        # Mixed Precision Training
        self.scaler = GradScaler() if config.use_amp else None

        # Early Stopping
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        )

        # TensorBoard
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # 체크포인트 디렉토리
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # 상태 추적
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_epoch = 0

        # 체크포인트 복원
        if config.resume_from_checkpoint:
            self.load_checkpoint(config.resume_from_checkpoint)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0
        total_note_loss = 0
        total_concentration_loss = 0
        total_volume_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.max_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # 데이터 준비
            text_input = {
                'input_ids': batch['text_input_ids'].to(self.device),
                'attention_mask': batch['text_attention_mask'].to(self.device)
            }
            target_notes = batch['target_notes'].to(self.device)
            target_concentrations = batch['target_concentrations'].to(self.device)
            target_volumes = batch['target_volumes'].to(self.device)
            seq_lengths = batch['sequence_length']

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                # 모델 입력
                inputs = {'text': text_input}
                targets = {
                    'notes': target_notes,
                    'concentrations': target_concentrations,
                    'volumes': target_volumes
                }

                # 예측
                outputs = self.model(inputs, targets)

                # Loss 계산
                note_logits = outputs['note_logits']  # (B, L, num_notes)
                concentration_logits = outputs['concentration_logits']  # (B, L, num_concentrations)
                volume_predictions = outputs['volume_predictions']  # (B, L)

                # Reshape for loss calculation
                B, L, C_notes = note_logits.shape
                _, _, C_conc = concentration_logits.shape

                note_loss = self.note_loss_fn(
                    note_logits.permute(0, 2, 1),  # (B, C, L)
                    target_notes
                )

                concentration_loss = self.concentration_loss_fn(
                    concentration_logits.permute(0, 2, 1),  # (B, C, L)
                    target_concentrations
                )

                volume_loss = self.volume_loss_fn(
                    volume_predictions,
                    target_volumes
                )

                # 총 loss (가중 평균)
                loss = note_loss + concentration_loss + volume_loss * 0.1

                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass with mixed precision
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # TensorBoard 로깅
                if self.global_step % self.config.log_every_n_steps == 0:
                    self.writer.add_scalar('train/loss', loss.item() * self.config.gradient_accumulation_steps, self.global_step)
                    self.writer.add_scalar('train/note_loss', note_loss.item(), self.global_step)
                    self.writer.add_scalar('train/concentration_loss', concentration_loss.item(), self.global_step)
                    self.writer.add_scalar('train/volume_loss', volume_loss.item(), self.global_step)
                    self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)

            # 통계 업데이트
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_note_loss += note_loss.item()
            total_concentration_loss += concentration_loss.item()
            total_volume_loss += volume_loss.item()

            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'note_loss': f'{note_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        # 에폭 평균
        n_batches = len(self.train_loader)
        return {
            'loss': total_loss / n_batches,
            'note_loss': total_note_loss / n_batches,
            'concentration_loss': total_concentration_loss / n_batches,
            'volume_loss': total_volume_loss / n_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """검증"""
        self.model.eval()
        total_loss = 0
        total_note_loss = 0
        total_concentration_loss = 0
        total_volume_loss = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            # 데이터 준비
            text_input = {
                'input_ids': batch['text_input_ids'].to(self.device),
                'attention_mask': batch['text_attention_mask'].to(self.device)
            }
            target_notes = batch['target_notes'].to(self.device)
            target_concentrations = batch['target_concentrations'].to(self.device)
            target_volumes = batch['target_volumes'].to(self.device)

            with autocast(enabled=self.config.use_amp):
                # 모델 입력
                inputs = {'text': text_input}
                targets = {
                    'notes': target_notes,
                    'concentrations': target_concentrations,
                    'volumes': target_volumes
                }

                # 예측
                outputs = self.model(inputs, targets)

                # Loss 계산
                note_logits = outputs['note_logits']
                concentration_logits = outputs['concentration_logits']
                volume_predictions = outputs['volume_predictions']

                note_loss = self.note_loss_fn(
                    note_logits.permute(0, 2, 1),
                    target_notes
                )

                concentration_loss = self.concentration_loss_fn(
                    concentration_logits.permute(0, 2, 1),
                    target_concentrations
                )

                volume_loss = self.volume_loss_fn(
                    volume_predictions,
                    target_volumes
                )

                loss = note_loss + concentration_loss + volume_loss * 0.1

            total_loss += loss.item()
            total_note_loss += note_loss.item()
            total_concentration_loss += concentration_loss.item()
            total_volume_loss += volume_loss.item()

        n_batches = len(self.val_loader)
        return {
            'loss': total_loss / n_batches,
            'note_loss': total_note_loss / n_batches,
            'concentration_loss': total_concentration_loss / n_batches,
            'volume_loss': total_volume_loss / n_batches
        }

    def train(self):
        """전체 학습 루프"""
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.max_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {self.config.use_amp}")
        logger.info("=" * 60)

        for epoch in range(self.start_epoch, self.config.max_epochs):
            # 학습
            train_metrics = self.train_epoch(epoch)

            # 스케줄러 업데이트
            self.scheduler.step()

            # 검증
            val_metrics = self.validate()

            # 로깅
            logger.info(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")

            self.writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)

            # Best model 저장
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"✓ New best model saved (val_loss: {val_metrics['loss']:.4f})")

            # 주기적 체크포인트
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break

        logger.info("\n" + "=" * 60)
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info("=" * 60)

        self.writer.close()

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Best model
        if is_best:
            path = Path(self.config.checkpoint_dir) / 'best_model.pt'
            torch.save(checkpoint, path)
            logger.info(f"Best checkpoint saved: {path}")
        else:
            # 일반 체크포인트
            path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")

            # 오래된 체크포인트 정리
            self.cleanup_old_checkpoints()

    def cleanup_old_checkpoints(self):
        """오래된 체크포인트 삭제"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            [f for f in checkpoint_dir.glob('checkpoint_epoch_*.pt')],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        # keep_last_n_checkpoints보다 많으면 삭제
        for checkpoint in checkpoints[self.config.keep_last_n_checkpoints:]:
            checkpoint.unlink()
            logger.info(f"Deleted old checkpoint: {checkpoint}")

    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        logger.info(f"Resumed from epoch {self.start_epoch}")


def load_data_from_db(session: Session, min_ingredients: int = 3) -> List[Recipe]:
    """데이터베이스에서 레시피 로드"""
    logger.info("Loading recipes from database...")

    # 재료가 충분한 레시피만 로드
    recipes = session.query(Recipe).join(Recipe.ingredients).group_by(Recipe.id).having(
        func.count(RecipeIngredient.id) >= min_ingredients
    ).all()

    logger.info(f"Loaded {len(recipes)} recipes")
    return recipes


def build_note_vocabulary(recipes: List[Recipe]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """향료 노트 vocabulary 구축"""
    logger.info("Building note vocabulary...")

    note_names = set()
    for recipe in recipes:
        for ingredient in recipe.ingredients:
            note_names.add(ingredient.note.name)

    # 특수 토큰 추가
    note_to_idx = {
        '[PAD]': 0,
        '[UNK]': 1,
        '[SOS]': 2,  # Start of sequence
        '[EOS]': 3   # End of sequence
    }

    # 노트 추가
    for idx, note in enumerate(sorted(note_names), start=4):
        note_to_idx[note] = idx

    idx_to_note = {v: k for k, v in note_to_idx.items()}

    logger.info(f"Vocabulary size: {len(note_to_idx)}")
    return note_to_idx, idx_to_note


def main(args):
    """메인 함수"""
    # 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 설정
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # 데이터 로드
    with get_session() as session:
        recipes = load_data_from_db(session, min_ingredients=3)

    if len(recipes) == 0:
        logger.error("No recipes found in database!")
        return

    # Vocabulary 구축
    note_to_idx, idx_to_note = build_note_vocabulary(recipes)

    # 모델 설정
    model_config = FragranceGenerationConfig()
    model_config.num_fragrance_notes = len(note_to_idx)
    model_config.max_recipe_length = training_config.max_sequence_length

    # 모델 생성
    logger.info("Creating model...")
    model = UniversalFragranceGenerator(model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained('klue/bert-base')

    # 데이터셋 생성
    logger.info("Creating datasets...")
    dataset = FragranceDataset(recipes, tokenizer, note_to_idx, training_config.max_sequence_length)

    # Train/Val/Test 분할
    n_total = len(dataset)
    n_train = int(n_total * training_config.train_split)
    n_val = int(n_total * training_config.val_split)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(args.seed)
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True
    )

    # Trainer 생성
    trainer = Trainer(model, train_loader, val_loader, training_config)

    # 학습 시작
    trainer.train()

    # Vocabulary 저장
    vocab_path = Path(training_config.checkpoint_dir) / 'note_vocabulary.json'
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump({
            'note_to_idx': note_to_idx,
            'idx_to_note': idx_to_note
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Vocabulary saved: {vocab_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UniversalFragranceGenerator")

    # 학습 설정
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')

    # 경로
    parser.add_argument('--checkpoint_dir', type=str,
                       default='checkpoints/universal_fragrance',
                       help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str,
                       default='runs/universal_fragrance',
                       help='TensorBoard log directory')

    # 복원
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Resume from checkpoint path')

    # 기타
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    main(args)
