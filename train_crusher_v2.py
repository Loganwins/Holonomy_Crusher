#!/usr/bin/env python3
"""
HOLONOMY CRUSHER TRAINING v2
============================

CRITICAL: The crusher MUST be trained for holonomy to correspond to
semantic inconsistency. Without training, it crushes randomly.

This script trains the connection on contrastive pairs:
- Consistent: valid arguments, proofs, coherent reasoning
- Inconsistent: fallacies, contradictions, non-sequiturs

CHANGES FROM v1:
1. More rigorous training data
2. Validation set to measure actual consistency correlation
3. Learning rate scheduling
4. Early stopping based on margin improvement
5. Proper train/val split

Author: Logan Napolitano
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import json
import warnings


# =============================================================================
# TRAINING DATA - EXPANDED AND CATEGORIZED
# =============================================================================

# Category 1: Classical Logic
LOGIC_CONSISTENT = [
    # Modus Ponens
    "If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
    "If Alice is human, Alice is mortal. Alice is human. Therefore, Alice is mortal.",
    "If the battery is dead, the car won't start. The battery is dead. Therefore, the car won't start.",
    
    # Modus Tollens
    "If it rains, the ground gets wet. The ground is not wet. Therefore, it is not raining.",
    "If he studied, he passed. He did not pass. Therefore, he did not study.",
    
    # Hypothetical Syllogism
    "If A then B. If B then C. Therefore, if A then C.",
    "If it rains, the streets are wet. If the streets are wet, driving is dangerous. Therefore, if it rains, driving is dangerous.",
    
    # Categorical Syllogism
    "All mammals are warm-blooded. All dogs are mammals. Therefore, all dogs are warm-blooded.",
    "No reptiles are mammals. All snakes are reptiles. Therefore, no snakes are mammals.",
    "All squares are rectangles. All rectangles are parallelograms. Therefore, all squares are parallelograms.",
]

LOGIC_INCONSISTENT = [
    # Affirming the Consequent
    "If it rains, the ground gets wet. The ground is wet. Therefore, it rained.",
    "If she studied, she passed. She passed. Therefore, she studied.",
    "If he's guilty, he's nervous. He's nervous. Therefore, he's guilty.",
    
    # Denying the Antecedent
    "If it rains, the ground gets wet. It is not raining. Therefore, the ground is not wet.",
    "If she's rich, she's happy. She's not rich. Therefore, she's not happy.",
    
    # Undistributed Middle
    "All dogs are mammals. All cats are mammals. Therefore, all dogs are cats.",
    "All humans need water. All plants need water. Therefore, all humans are plants.",
    
    # Non Sequitur
    "The sky is blue. Therefore, capitalism is the best economic system.",
    "I had eggs for breakfast. Therefore, the stock market will rise today.",
]

# Category 2: Mathematical Reasoning
MATH_CONSISTENT = [
    "Given x = 3 and y = 4, we have x + y = 7. Since 7 > 5, we conclude x + y > 5.",
    "If a triangle has sides 3, 4, 5, then 3² + 4² = 9 + 16 = 25 = 5². By the Pythagorean theorem, this is a right triangle.",
    "The sum 1 + 2 + ... + n = n(n+1)/2. For n = 10, sum = 10 × 11 / 2 = 55.",
    "If n is even, then n = 2k for some integer k. Then n² = 4k², which is divisible by 4.",
    "Let f(x) = x². Then f(3) = 9 and f(-3) = 9. So f(3) = f(-3).",
    "If a > b and b > c, then a > c. Given 5 > 3 and 3 > 1, we have 5 > 1.",
]

MATH_INCONSISTENT = [
    "Given x = 2, we have x² = 4. Therefore x = 4.",
    "2 + 2 = 4. 4 = 2 + 2. Therefore 2 = 4.",
    "If n² = 4, then n = 2. But (-2)² = 4, so -2 = 2.",
    "The area of a circle is πr². The circumference is 2πr. Therefore area equals circumference.",
    "1/2 = 0.5 and 2/4 = 0.5. Therefore 1 = 2 and 2 = 4.",
    "x + 1 = 2 means x = 1. x + 2 = 3 means x = 1. Therefore 1 + 1 = 1 + 2.",
]

# Category 3: Causal Reasoning
CAUSAL_CONSISTENT = [
    "The plant died because it was not watered. Water is necessary for plant survival. Without water, the plant could not survive.",
    "She studied hard for the exam. As a result, she understood the material well. Because she understood the material, she passed the exam.",
    "The ice melted because the temperature rose above freezing. Above freezing, water exists as liquid, not ice.",
    "The circuit failed because a wire was disconnected. Current cannot flow through a disconnected wire. Without current, the circuit cannot function.",
    "He was late because he missed the bus. Missing the bus meant he had to walk. Walking takes longer than the bus.",
]

CAUSAL_INCONSISTENT = [
    "The rooster crowed. Then the sun rose. Therefore, the rooster's crow caused the sunrise.",
    "She wore her lucky socks. She won the game. Therefore, the socks caused her to win.",
    "Ice cream sales increase in summer. Drowning deaths increase in summer. Therefore, ice cream causes drowning.",
    "Countries with more Nobel laureates have higher chocolate consumption. Therefore, chocolate causes scientific genius.",
    "He prayed for rain. It rained. Therefore, his prayer caused the rain.",
]

# Category 4: Self-Contradiction
SELF_CONSISTENT = [
    "This statement contains five words. Let me count: this, statement, contains, five, words. Yes, exactly five.",
    "I know that I know nothing with certainty. This epistemic humility is itself a form of knowledge.",
    "All generalizations have exceptions, including this one. This is not a contradiction—it's self-referentially consistent.",
]

SELF_INCONSISTENT = [
    "This statement is false. If it's true, it's false. If it's false, it's true.",
    "I always lie. This statement is true. Therefore, I sometimes tell the truth, which means I don't always lie.",
    "Nothing is absolute. This absolute statement proves itself wrong.",
    "There are no rules. This rule proves there is at least one rule.",
    "I know everything. I don't know if this statement is true. Therefore, I don't know everything.",
]

# Category 5: Narrative Coherence
NARRATIVE_CONSISTENT = [
    "John entered the room. He turned on the light. Now he could see clearly. He found his keys on the table.",
    "She opened the book to chapter one. She read the first page. She turned to the second page. She continued reading.",
    "The train departed at 9 AM. It traveled for two hours. It arrived at 11 AM.",
    "He planted the seed in spring. It grew through summer. He harvested the crop in fall.",
]

NARRATIVE_INCONSISTENT = [
    "John was alone in the locked room. Then he talked to Mary who was in the room. But no one else was in the room.",
    "She read chapter 5, then chapter 3, then chapter 7, finishing the book in order.",
    "The train left at 9 AM and arrived at 8 AM the same day without time travel.",
    "He harvested the crop, then planted the seed, then the crop grew.",
    "It was a dark night. The sun was shining brightly. There were no lights but everything was clearly visible.",
]


def get_all_training_data() -> Tuple[List[str], List[str]]:
    """Combine all training data."""
    consistent = (
        LOGIC_CONSISTENT + 
        MATH_CONSISTENT + 
        CAUSAL_CONSISTENT + 
        SELF_CONSISTENT + 
        NARRATIVE_CONSISTENT
    )
    
    inconsistent = (
        LOGIC_INCONSISTENT + 
        MATH_INCONSISTENT + 
        CAUSAL_INCONSISTENT + 
        SELF_INCONSISTENT + 
        NARRATIVE_INCONSISTENT
    )
    
    return consistent, inconsistent


# =============================================================================
# DATASET
# =============================================================================

class ConsistencyDataset(Dataset):
    """Dataset of consistent/inconsistent text pairs."""
    
    def __init__(self, 
                 consistent: List[str], 
                 inconsistent: List[str],
                 tokenizer,
                 max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Balance dataset
        min_len = min(len(consistent), len(inconsistent))
        self.consistent = consistent[:min_len]
        self.inconsistent = inconsistent[:min_len]
        
        # Pair them up
        self.pairs = list(zip(self.consistent, self.inconsistent))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        c_text, i_text = self.pairs[idx]
        
        c_tokens = self.tokenizer(
            c_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        i_tokens = self.tokenizer(
            i_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'consistent_ids': c_tokens['input_ids'].squeeze(),
            'inconsistent_ids': i_tokens['input_ids'].squeeze(),
            'consistent_mask': c_tokens['attention_mask'].squeeze(),
            'inconsistent_mask': i_tokens['attention_mask'].squeeze(),
        }


# =============================================================================
# TRAINER v2
# =============================================================================

class CrusherTrainer:
    """
    Trains the crusher's connection to align holonomy with semantic inconsistency.
    
    The goal: after training, high holonomy should correlate with actual
    logical/semantic inconsistency, not arbitrary geometric movement.
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 crusher,
                 lr: float = 1e-4,
                 weight_decay: float = 0.01,
                 device: str = 'cuda'):
        
        self.model = model
        self.tokenizer = tokenizer
        self.crusher = crusher
        self.device = device
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            crusher.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize margin
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # History
        self.history = {
            'train_loss': [],
            'train_margin': [],
            'val_loss': [],
            'val_margin': [],
            'lr': [],
        }
        
        # Best model tracking
        self.best_margin = -float('inf')
        self.best_state = None
        self.patience_counter = 0
    
    def encode_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        """Encode a batch of text into geometric states."""
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            hidden = outputs.hidden_states[-1]
        
        # Get non-padding positions
        batch_states = []
        for b in range(hidden.shape[0]):
            mask = attention_mask[b].bool()
            valid_hidden = hidden[b, mask]
            
            # Encode each valid position
            states = []
            for i in range(valid_hidden.shape[0]):
                state = self.crusher.encode_state(valid_hidden[i:i+1])
                states.append(state.squeeze())
            
            batch_states.append(states)
        
        return batch_states
    
    def compute_batch_holonomy(self, batch_states: List[List[torch.Tensor]]) -> torch.Tensor:
        """Compute average holonomy for a batch."""
        holonomies = []
        
        for states in batch_states:
            if len(states) >= 3:
                # Use sliding windows
                window_hols = []
                for i in range(len(states) - 2):
                    window = states[i:i+3]
                    hol = self.crusher.holonomy_calc.compute_loop_holonomy(window)
                    window_hols.append(hol)
                
                if window_hols:
                    avg_hol = sum(window_hols) / len(window_hols)
                    holonomies.append(avg_hol)
        
        if holonomies:
            return torch.stack(holonomies).mean()
        return torch.tensor(0.0, device=self.device)
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training step."""
        self.optimizer.zero_grad()
        
        # Encode consistent and inconsistent
        c_states = self.encode_batch(
            batch['consistent_ids'],
            batch['consistent_mask']
        )
        i_states = self.encode_batch(
            batch['inconsistent_ids'],
            batch['inconsistent_mask']
        )
        
        # Compute holonomies (need gradients for inconsistent)
        with torch.no_grad():
            hol_c = self.compute_batch_holonomy(c_states)
        
        # For inconsistent, we want gradients
        hol_i = self.compute_batch_holonomy(i_states)
        
        # Contrastive loss: hol_c should be MUCH LESS than hol_i
        margin = 0.2  # Target margin
        contrastive_loss = F.relu(hol_c - hol_i + margin)
        
        # Regularization: don't let holonomies explode
        reg_loss = 0.01 * (hol_c + hol_i)
        
        # Separation loss: push inconsistent holonomy higher
        separation_loss = F.relu(0.1 - hol_i)  # Want hol_i > 0.1
        
        # Total loss
        loss = contrastive_loss + reg_loss + separation_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.crusher.parameters(), 1.0)
        self.optimizer.step()
        
        # Compute margin
        margin_achieved = (hol_i - hol_c).item()
        
        return {
            'loss': loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'reg_loss': reg_loss.item(),
            'separation_loss': separation_loss.item(),
            'hol_consistent': hol_c.item(),
            'hol_inconsistent': hol_i.item(),
            'margin': margin_achieved,
        }
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict:
        """Validate on held-out data."""
        total_loss = 0
        total_margin = 0
        n_batches = 0
        
        for batch in dataloader:
            c_states = self.encode_batch(
                batch['consistent_ids'],
                batch['consistent_mask']
            )
            i_states = self.encode_batch(
                batch['inconsistent_ids'],
                batch['inconsistent_mask']
            )
            
            hol_c = self.compute_batch_holonomy(c_states)
            hol_i = self.compute_batch_holonomy(i_states)
            
            margin = 0.2
            loss = F.relu(hol_c - hol_i + margin)
            
            total_loss += loss.item()
            total_margin += (hol_i - hol_c).item()
            n_batches += 1
        
        return {
            'val_loss': total_loss / n_batches if n_batches > 0 else 0,
            'val_margin': total_margin / n_batches if n_batches > 0 else 0,
        }
    
    def train_epoch(self, 
                    train_loader: DataLoader,
                    val_loader: DataLoader = None,
                    verbose: bool = True) -> Dict:
        """Train for one epoch."""
        self.crusher.train()
        
        epoch_metrics = {
            'loss': [],
            'margin': [],
            'hol_consistent': [],
            'hol_inconsistent': [],
        }
        
        iterator = tqdm(train_loader, desc="Training") if verbose else train_loader
        
        for batch in iterator:
            metrics = self.train_step(batch)
            
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k].append(metrics[k])
            
            if verbose:
                iterator.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'margin': f"{metrics['margin']:.4f}",
                })
        
        # Average metrics
        avg_metrics = {k: sum(v)/len(v) for k, v in epoch_metrics.items() if v}
        
        # Validation
        if val_loader is not None:
            self.crusher.eval()
            val_metrics = self.validate(val_loader)
            avg_metrics.update(val_metrics)
            
            # Update scheduler based on validation margin
            self.scheduler.step(val_metrics['val_margin'])
            
            # Check for improvement
            if val_metrics['val_margin'] > self.best_margin:
                self.best_margin = val_metrics['val_margin']
                self.best_state = {k: v.clone() for k, v in self.crusher.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        # Record history
        self.history['train_loss'].append(avg_metrics.get('loss', 0))
        self.history['train_margin'].append(avg_metrics.get('margin', 0))
        self.history['val_loss'].append(avg_metrics.get('val_loss', 0))
        self.history['val_margin'].append(avg_metrics.get('val_margin', 0))
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
        
        return avg_metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader = None,
              epochs: int = 50,
              early_stopping_patience: int = 10,
              verbose: bool = True) -> Dict:
        """Full training loop with early stopping."""
        
        if verbose:
            print("=" * 60)
            print("HOLONOMY CRUSHER TRAINING")
            print("=" * 60)
            print(f"Epochs: {epochs}")
            print(f"Early stopping patience: {early_stopping_patience}")
            print(f"Train batches: {len(train_loader)}")
            if val_loader:
                print(f"Val batches: {len(val_loader)}")
            print("=" * 60)
        
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader, val_loader, verbose)
            
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}:")
                print(f"  Train Loss: {metrics.get('loss', 0):.4f}")
                print(f"  Train Margin: {metrics.get('margin', 0):.4f}")
                if val_loader:
                    print(f"  Val Loss: {metrics.get('val_loss', 0):.4f}")
                    print(f"  Val Margin: {metrics.get('val_margin', 0):.4f}")
                print(f"  Hol(consistent): {metrics.get('hol_consistent', 0):.4f}")
                print(f"  Hol(inconsistent): {metrics.get('hol_inconsistent', 0):.4f}")
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # Restore best model
        if self.best_state is not None:
            self.crusher.load_state_dict(self.best_state)
            if verbose:
                print(f"\nRestored best model (margin: {self.best_margin:.4f})")
        
        # Mark connection as trained
        self.crusher.holonomy_calc.connection.mark_trained()
        
        return self.history
    
    def save(self, path: str):
        """Save trained crusher."""
        torch.save({
            'crusher_state_dict': self.crusher.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_margin': self.best_margin,
        }, path)
        print(f"Saved to {path}")
    
    def load(self, path: str):
        """Load trained crusher."""
        checkpoint = torch.load(path, map_location=self.device)
        self.crusher.load_state_dict(checkpoint['crusher_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        self.best_margin = checkpoint.get('best_margin', self.best_margin)
        self.crusher.holonomy_calc.connection.mark_trained()
        print(f"Loaded from {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Holonomy Crusher v2')
    parser.add_argument('--model', type=str, default='NousResearch/Hermes-3-Llama-3.1-8B')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--output', type=str, default='crusher_trained.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    print("=" * 60)
    print("HOLONOMY CRUSHER TRAINING v2")
    print("=" * 60)
    
    # Get training data
    consistent, inconsistent = get_all_training_data()
    print(f"\nTraining data:")
    print(f"  Consistent examples: {len(consistent)}")
    print(f"  Inconsistent examples: {len(inconsistent)}")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"\nLoading model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # For demo, use dummy model if full model unavailable
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map=args.device,
                torch_dtype=torch.float16,
            )
            d_model = model.config.hidden_size
            print(f"  Loaded on {args.device}")
        except Exception as e:
            print(f"  Could not load full model: {e}")
            print("  Using dummy model for demo...")
            
            class DummyConfig:
                hidden_size = 512
            
            class DummyModel(nn.Module):
                config = DummyConfig()
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(32000, 512)
                def forward(self, input_ids, **kwargs):
                    hidden = self.embed(input_ids)
                    class Output:
                        hidden_states = [hidden]
                    return Output()
            
            model = DummyModel()
            d_model = 512
            args.device = 'cpu'
        
        # Create dataset
        dataset = ConsistencyDataset(consistent, inconsistent, tokenizer)
        
        # Split
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        print(f"\nDataset split:")
        print(f"  Train: {train_size}")
        print(f"  Val: {val_size}")
        
        # Create crusher
        from holonomy_crusher_v2 import HolonomyCrusher, CrusherConfig, CrushMode, RecoveryMode
        
        config = CrusherConfig(
            d_model=d_model,
            d_fiber=64,
            crush_lambda=50.0,
            crush_mode=CrushMode.HARD,
            recovery_mode=RecoveryMode.BACKTRACK,
        )
        
        crusher = HolonomyCrusher(config).to(args.device)
        
        # Create trainer
        trainer = CrusherTrainer(
            model=model,
            tokenizer=tokenizer,
            crusher=crusher,
            lr=args.lr,
            device=args.device
        )
        
        # Train
        history = trainer.train(
            train_loader,
            val_loader,
            epochs=args.epochs,
            early_stopping_patience=10
        )
        
        # Save
        trainer.save(args.output)
        
        # Summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best validation margin: {trainer.best_margin:.4f}")
        print(f"Final train margin: {history['train_margin'][-1]:.4f}")
        print(f"\nThe connection is now trained!")
        print("Holonomy should correlate with semantic inconsistency.")
        print(f"\nWeights saved to: {args.output}")
        
    except ImportError as e:
        print(f"\nMissing dependencies: {e}")
        print("Install with: pip install transformers torch")


if __name__ == "__main__":
    main()
