# UniGaze Research Improvements Checklist for Journal Publication

**Research Focus:** Universal Gaze Estimation via Large-scale Pre-Training  
**Target:** International Computer Vision/ML Journals (CVPR, ICCV, ECCV, TPAMI, IJCV, etc.)  
**Date:** 2025-01-27

---

## Executive Summary

This document identifies **core functionality gaps** and **research improvement opportunities** in the UniGaze codebase that can lead to significant contributions for journal publication. The focus is on **methodological innovations**, **experimental validation**, and **theoretical contributions** rather than code quality improvements.

---

## 1. CURRENT CORE FUNCTIONALITY ANALYSIS

### 1.1 Model Architecture (`unigaze/models/vit/mae_gaze.py`)

**Current Implementation:**
- ✅ MAE pre-trained ViT backbone (B16, L16, H14)
- ✅ Simple linear head: `nn.Linear(embed_dim, 2)` for pitch/yaw prediction
- ✅ Global pooling from ViT features
- ✅ Direct regression to gaze angles

**Research Limitations:**
1. **Oversimplified Head Architecture**
   - Single linear layer may not capture complex gaze patterns
   - No multi-scale feature fusion
   - No attention mechanism in prediction head
   - Missing intermediate representations

2. **Feature Extraction Limitations**
   - Only uses global pooled features
   - No spatial attention to eye regions
   - No multi-resolution feature fusion
   - Missing temporal information for video sequences

3. **No Uncertainty Estimation**
   - Cannot quantify prediction confidence
   - No epistemic/aleatoric uncertainty modeling
   - Missing reliability metrics

---

### 1.2 Loss Functions (`unigaze/criteria/gaze_loss.py`)

**Current Implementation:**
- ✅ L1 Loss (Mean Absolute Error)
- ✅ L2 Loss (Mean Squared Error)
- ✅ Simple pitch/yaw regression

**Research Limitations:**
1. **Geometric Inconsistency**
   - L1/L2 on pitch/yaw doesn't respect spherical geometry
   - Should use angular loss on 3D gaze vectors
   - Missing geometric constraints

2. **No Multi-Task Learning**
   - Only predicts gaze, ignores head pose
   - Could jointly learn gaze + head pose + landmarks
   - Missing auxiliary tasks for better representation

3. **No Adaptive Loss Weighting**
   - Fixed loss weights
   - No curriculum learning
   - Missing difficulty-aware sampling

4. **Limited Loss Functions**
   - No cosine similarity loss
   - No angular error loss
   - No contrastive learning for gaze similarity

---

### 1.3 Training Strategy (`unigaze/trainers/simple_trainer.py`)

**Current Implementation:**
- ✅ Standard supervised learning
- ✅ Joint dataset training
- ✅ Cross-dataset evaluation
- ✅ Basic data augmentation

**Research Limitations:**
1. **No Domain Adaptation**
   - No explicit domain adaptation techniques
   - Missing adversarial domain adaptation
   - No domain-invariant feature learning

2. **Limited Data Augmentation**
   - Basic ImageNet-style augmentation
   - No gaze-aware augmentation
   - Missing synthetic data generation

3. **No Few-Shot Learning**
   - Cannot adapt to new subjects quickly
   - Missing meta-learning approaches
   - No personalization mechanisms

4. **No Active Learning**
   - Cannot identify hard samples
   - Missing uncertainty-based sampling
   - No curriculum learning strategies

---

### 1.4 Pre-training Strategy (`MAE/`)

**Current Implementation:**
- ✅ Standard MAE pre-training
- ✅ Masked image reconstruction
- ✅ Multiple face datasets

**Research Limitations:**
1. **No Gaze-Aware Pre-training**
   - MAE doesn't use gaze information during pre-training
   - Could use gaze-conditional masking
   - Missing gaze-guided reconstruction

2. **Limited Pre-training Tasks**
   - Only reconstruction task
   - Could add contrastive learning
   - Missing multi-task pre-training

3. **No Progressive Pre-training**
   - Fixed mask ratio (0.75)
   - Could use curriculum masking
   - Missing adaptive masking strategies

---

## 2. RESEARCH IMPROVEMENT OPPORTUNITIES

### 2.1 HIGH PRIORITY: Novel Methodological Contributions

#### ✅ **Improvement 1: Geometric-Aware Loss Functions**
**Research Gap:** Current L1/L2 loss doesn't respect spherical geometry of gaze

**Proposed Solution:**
```python
# Implement angular loss on 3D gaze vectors
class AngularGazeLoss(nn.Module):
    def forward(self, pred_pitchyaw, gt_pitchyaw):
        # Convert to 3D vectors
        pred_vec = pitchyaw_to_vector(pred_pitchyaw)
        gt_vec = pitchyaw_to_vector(gt_pitchyaw)
        # Compute angular error
        cos_sim = F.cosine_similarity(pred_vec, gt_vec)
        angular_error = torch.acos(torch.clamp(cos_sim, -1+1e-7, 1-1e-7))
        return angular_error.mean()
```

**Expected Contribution:**
- More geometrically consistent training
- Better generalization to extreme gaze angles
- Theoretical justification for spherical geometry

**Implementation Checklist:**
- [ ] Implement `AngularGazeLoss` class
- [ ] Add `pitchyaw_to_vector` utility function
- [ ] Compare with L1/L2 baselines
- [ ] Ablation study on loss functions
- [ ] Theoretical analysis of geometric properties

---

#### ✅ **Improvement 2: Multi-Scale Feature Fusion Head**
**Research Gap:** Simple linear head may lose spatial information

**Proposed Solution:**
```python
class MultiScaleGazeHead(nn.Module):
    def __init__(self, embed_dim, num_layers=3):
        # Multi-scale feature extraction
        self.attention_pool = nn.MultiheadAttention(embed_dim, num_heads=8)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, 2)
        )
    
    def forward(self, features):
        # features: [B, N, D] from ViT
        # Apply attention pooling
        pooled, attn_weights = self.attention_pool(
            features.mean(dim=1, keepdim=True), features, features
        )
        return self.mlp(pooled.squeeze(1)), attn_weights
```

**Expected Contribution:**
- Better feature utilization
- Interpretable attention maps
- Improved accuracy on challenging cases

**Implementation Checklist:**
- [ ] Design multi-scale head architecture
- [ ] Implement attention-based pooling
- [ ] Add feature pyramid fusion
- [ ] Visualize attention maps
- [ ] Compare with baseline linear head
- [ ] Ablation on architecture components

---

#### ✅ **Improvement 3: Uncertainty-Aware Gaze Estimation**
**Research Gap:** No confidence/uncertainty estimation

**Proposed Solution:**
```python
class UncertaintyGazeHead(nn.Module):
    def __init__(self, embed_dim):
        self.gaze_head = nn.Linear(embed_dim, 2)
        self.uncertainty_head = nn.Linear(embed_dim, 2)  # Aleatoric uncertainty
    
    def forward(self, features):
        gaze_pred = self.gaze_head(features)
        log_var = self.uncertainty_head(features)
        return {
            'pred_gaze': gaze_pred,
            'log_var': log_var,
            'uncertainty': torch.exp(log_var)
        }

# Loss with uncertainty
def uncertainty_loss(pred, gt, log_var):
    precision = torch.exp(-log_var)
    return (precision * (pred - gt)**2 + log_var).mean()
```

**Expected Contribution:**
- Quantify prediction reliability
- Enable active learning
- Better failure detection
- Novel evaluation metric

**Implementation Checklist:**
- [ ] Implement uncertainty estimation head
- [ ] Design uncertainty-aware loss
- [ ] Add epistemic uncertainty (Monte Carlo dropout)
- [ ] Evaluate uncertainty calibration
- [ ] Use uncertainty for active learning
- [ ] Compare uncertainty with actual errors

---

#### ✅ **Improvement 4: Gaze-Aware MAE Pre-training**
**Research Gap:** MAE doesn't leverage gaze information during pre-training

**Proposed Solution:**
```python
class GazeConditionalMAE(MaskedAutoencoderViT):
    def forward(self, imgs, gaze_labels, mask_ratio=0.75):
        # Gaze-guided masking: mask regions away from eyes
        eye_region_mask = self.get_eye_region_mask(imgs, gaze_labels)
        mask = self.random_masking_with_guidance(eye_region_mask, mask_ratio)
        
        # Gaze-conditional reconstruction
        latent, _, ids_restore = self.forward_encoder(imgs, mask)
        pred = self.forward_decoder(latent, ids_restore, gaze_labels)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
```

**Expected Contribution:**
- Better pre-training for gaze estimation
- Domain-specific representation learning
- Novel pre-training paradigm

**Implementation Checklist:**
- [ ] Design gaze-conditional masking
- [ ] Implement gaze-guided reconstruction
- [ ] Compare with standard MAE pre-training
- [ ] Ablation on masking strategies
- [ ] Analyze learned representations

---

#### ✅ **Improvement 5: Domain Adaptation Framework**
**Research Gap:** No explicit domain adaptation for cross-dataset generalization

**Proposed Solution:**
```python
class DomainAdaptiveGaze(nn.Module):
    def __init__(self, backbone, num_domains):
        self.backbone = backbone
        self.gaze_head = nn.Linear(embed_dim, 2)
        self.domain_classifier = nn.Linear(embed_dim, num_domains)
    
    def forward(self, x, alpha=1.0, mode='train'):
        features = self.backbone(x)
        gaze_pred = self.gaze_head(features)
        
        if mode == 'train':
            # Gradient reversal for domain adaptation
            reversed_features = GradientReversal.apply(features, alpha)
            domain_pred = self.domain_classifier(reversed_features)
            return {'pred_gaze': gaze_pred, 'domain_pred': domain_pred}
        return {'pred_gaze': gaze_pred}
```

**Expected Contribution:**
- Better cross-dataset generalization
- Domain-invariant features
- Novel adaptation strategy

**Implementation Checklist:**
- [ ] Implement gradient reversal layer
- [ ] Design domain classifier
- [ ] Multi-domain training setup
- [ ] Cross-dataset evaluation protocol
- [ ] Compare with baseline methods
- [ ] Analyze domain-invariant features

---

### 2.2 MEDIUM PRIORITY: Enhanced Training Strategies

#### ✅ **Improvement 6: Multi-Task Learning**
**Proposed:** Joint learning of gaze + head pose + eye landmarks

**Implementation Checklist:**
- [ ] Design multi-task head architecture
- [ ] Implement task-specific losses
- [ ] Add task weighting mechanism
- [ ] Evaluate task correlation
- [ ] Compare with single-task baseline

---

#### ✅ **Improvement 7: Contrastive Learning for Gaze**
**Proposed:** Learn gaze similarity in embedding space

**Implementation Checklist:**
- [ ] Design contrastive loss for gaze
- [ ] Implement positive/negative sampling
- [ ] Add projection head
- [ ] Evaluate learned representations
- [ ] Compare with supervised baseline

---

#### ✅ **Improvement 8: Curriculum Learning**
**Proposed:** Progressive training from easy to hard samples

**Implementation Checklist:**
- [ ] Design difficulty scoring function
- [ ] Implement curriculum scheduler
- [ ] Evaluate training dynamics
- [ ] Compare with random sampling

---

### 2.3 LOW PRIORITY: Advanced Features

#### ✅ **Improvement 9: Temporal Modeling for Video**
**Proposed:** Use temporal information in video sequences

**Implementation Checklist:**
- [ ] Design temporal encoder (LSTM/Transformer)
- [ ] Implement temporal consistency loss
- [ ] Evaluate on video datasets
- [ ] Compare with frame-by-frame baseline

---

#### ✅ **Improvement 10: Few-Shot Personalization**
**Proposed:** Adapt to new subjects with few samples

**Implementation Checklist:**
- [ ] Implement meta-learning framework
- [ ] Design few-shot adaptation strategy
- [ ] Evaluate on new subjects
- [ ] Compare with fine-tuning baseline

---

## 3. EXPERIMENTAL VALIDATION GAPS

### 3.1 Missing Evaluation Metrics

**Current:** Only angular error (mean/std)

**Needed:**
- [ ] **Per-subject analysis** - Individual subject performance
- [ ] **Per-angle analysis** - Performance across gaze angles
- [ ] **Failure case analysis** - When does model fail?
- [ ] **Computational efficiency** - FLOPs, latency, memory
- [ ] **Robustness metrics** - Occlusion, lighting, pose variations
- [ ] **Cross-dataset metrics** - Domain gap quantification
- [ ] **Statistical significance** - Multiple runs, confidence intervals

---

### 3.2 Missing Ablation Studies

**Needed:**
- [ ] **Pre-training ablation** - With/without MAE, different mask ratios
- [ ] **Architecture ablation** - Head design, feature fusion
- [ ] **Loss function ablation** - L1 vs L2 vs Angular
- [ ] **Data augmentation ablation** - Impact of each augmentation
- [ ] **Training strategy ablation** - Learning rate, scheduler, optimizer
- [ ] **Dataset contribution** - Which datasets help most?

---

### 3.3 Missing Baselines

**Needed:**
- [ ] **State-of-the-art comparisons** - FullGaze, Gaze360, etc.
- [ ] **Ablation baselines** - Without pre-training, different backbones
- [ ] **Cross-dataset baselines** - Domain adaptation methods
- [ ] **Efficiency baselines** - Model size vs accuracy trade-offs

---

### 3.4 Missing Analysis

**Needed:**
- [ ] **Attention visualization** - What does model focus on?
- [ ] **Feature visualization** - t-SNE of learned features
- [ ] **Error analysis** - Systematic error patterns
- [ ] **Failure mode analysis** - When and why it fails
- [ ] **Generalization analysis** - In-domain vs out-of-domain
- [ ] **Scalability analysis** - Performance vs dataset size

---

## 4. THEORETICAL CONTRIBUTIONS

### 4.1 Theoretical Analysis Needed

- [ ] **Convergence analysis** - Training dynamics
- [ ] **Generalization bounds** - Theoretical guarantees
- [ ] **Representation learning theory** - Why MAE helps?
- [ ] **Domain adaptation theory** - Transfer learning bounds
- [ ] **Uncertainty quantification theory** - Calibration guarantees

---

## 5. DATASET & EVALUATION IMPROVEMENTS

### 5.1 New Datasets Needed

- [ ] **Diverse demographic dataset** - Age, ethnicity, gender balance
- [ ] **Challenging conditions** - Extreme lighting, poses, occlusions
- [ ] **Real-world scenarios** - In-the-wild data
- [ ] **Long-range gaze** - Far distances, small faces
- [ ] **Multi-person scenarios** - Multiple people in frame

---

### 5.2 Evaluation Protocols

- [ ] **Standardized evaluation** - Consistent train/test splits
- [ ] **Cross-dataset protocols** - Fair comparison framework
- [ ] **Subject-independent evaluation** - Leave-one-subject-out
- [ ] **Temporal evaluation** - Video sequence protocols
- [ ] **Real-time evaluation** - Latency and throughput metrics

---

## 6. IMPLEMENTATION ROADMAP FOR JOURNAL SUBMISSION

### Phase 1: Core Methodological Improvements (Months 1-2)

**Priority 1: Loss Functions**
- [ ] Week 1-2: Implement angular loss
- [ ] Week 3-4: Implement uncertainty-aware loss
- [ ] Week 5-6: Ablation study on loss functions
- [ ] Week 7-8: Write-up and analysis

**Priority 2: Architecture**
- [ ] Week 1-2: Design multi-scale head
- [ ] Week 3-4: Implement attention mechanisms
- [ ] Week 5-6: Architecture ablation
- [ ] Week 7-8: Visualization and analysis

**Deliverables:**
- Improved model with new loss + architecture
- Ablation studies
- Performance improvements (target: 5-10% reduction in angular error)

---

### Phase 2: Advanced Training Strategies (Months 3-4)

**Priority 3: Pre-training**
- [ ] Week 1-2: Implement gaze-aware MAE
- [ ] Week 3-4: Compare with standard MAE
- [ ] Week 5-6: Analysis of learned representations

**Priority 4: Domain Adaptation**
- [ ] Week 1-2: Implement domain adaptation framework
- [ ] Week 3-4: Cross-dataset evaluation
- [ ] Week 5-6: Domain gap analysis

**Deliverables:**
- Novel pre-training method
- Domain adaptation framework
- Cross-dataset generalization improvements

---

### Phase 3: Comprehensive Evaluation (Months 5-6)

**Priority 5: Experiments**
- [ ] Week 1-4: Comprehensive evaluation on all datasets
- [ ] Week 5-6: Comparison with state-of-the-art
- [ ] Week 7-8: Ablation studies
- [ ] Week 9-10: Failure analysis and visualization
- [ ] Week 11-12: Statistical analysis and significance tests

**Deliverables:**
- Complete experimental results
- Comparison tables
- Visualization and analysis
- Statistical validation

---

### Phase 4: Paper Writing (Months 7-8)

**Priority 6: Documentation**
- [ ] Week 1-2: Related work review
- [ ] Week 3-4: Method section
- [ ] Week 5-6: Experiments section
- [ ] Week 7-8: Results, discussion, conclusion

**Deliverables:**
- Complete paper draft
- Supplementary material
- Code release
- Video demo

---

## 7. EXPECTED RESEARCH CONTRIBUTIONS

### 7.1 Novel Methodological Contributions

1. **Geometric-Aware Loss Functions**
   - First to use angular loss for gaze estimation
   - Theoretical analysis of geometric properties
   - Significant accuracy improvements

2. **Uncertainty-Aware Gaze Estimation**
   - Novel uncertainty quantification framework
   - Enables active learning and failure detection
   - New evaluation metrics

3. **Gaze-Aware Pre-training**
   - First gaze-conditional MAE
   - Better representation learning
   - Improved transfer learning

4. **Multi-Scale Attention Head**
   - Interpretable attention mechanisms
   - Better feature utilization
   - Visualization insights

5. **Domain Adaptation Framework**
   - Cross-dataset generalization
   - Domain-invariant features
   - Practical deployment improvements

---

### 7.2 Experimental Contributions

1. **Comprehensive Evaluation**
   - Largest-scale evaluation on multiple datasets
   - Cross-dataset generalization analysis
   - Real-world deployment validation

2. **Ablation Studies**
   - Systematic analysis of components
   - Design choices justification
   - Best practices for gaze estimation

3. **Failure Analysis**
   - Systematic error patterns
   - Failure mode identification
   - Improvement directions

---

### 7.3 Theoretical Contributions

1. **Representation Learning Analysis**
   - Why MAE helps gaze estimation?
   - Feature learning dynamics
   - Transfer learning insights

2. **Generalization Analysis**
   - Cross-dataset bounds
   - Domain gap quantification
   - Theoretical guarantees

---

## 8. PUBLICATION STRATEGY

### 8.1 Target Venues

**Tier 1 (Top-tier):**
- CVPR, ICCV, ECCV (Computer Vision)
- NeurIPS, ICML (Machine Learning)
- TPAMI, IJCV (Journals)

**Tier 2 (Strong):**
- WACV (Current venue - improvement paper)
- BMVC, ACCV
- CVIU, Image and Vision Computing

**Strategy:**
- Start with Tier 1, have Tier 2 as backup
- Consider workshop papers for preliminary results
- Build reputation with incremental improvements

---

### 8.2 Paper Structure

1. **Abstract** - Clear contribution statement
2. **Introduction** - Problem motivation, contributions
3. **Related Work** - Comprehensive survey
4. **Method** - Detailed technical description
5. **Experiments** - Comprehensive evaluation
6. **Analysis** - Ablation, visualization, failure analysis
7. **Conclusion** - Summary and future work

---

## 9. SUCCESS METRICS

### 9.1 Performance Targets

- [ ] **Accuracy:** 10-15% reduction in angular error vs baseline
- [ ] **Generalization:** 20% improvement in cross-dataset performance
- [ ] **Efficiency:** Maintain or improve inference speed
- [ ] **Robustness:** Better performance on challenging conditions

### 9.2 Research Impact Targets

- [ ] **Novelty:** 3-5 novel methodological contributions
- [ ] **Completeness:** Comprehensive evaluation on 5+ datasets
- [ ] **Reproducibility:** Full code and model release
- [ ] **Clarity:** Clear presentation of contributions

---

## 10. IMMEDIATE ACTION ITEMS

### Week 1-2: Foundation
- [ ] Implement `AngularGazeLoss` class
- [ ] Implement `MultiScaleGazeHead` architecture
- [ ] Set up evaluation framework with new metrics
- [ ] Create baseline comparison script

### Week 3-4: Core Improvements
- [ ] Train models with new loss functions
- [ ] Train models with new architectures
- [ ] Run initial ablation studies
- [ ] Document initial results

### Week 5-6: Analysis
- [ ] Compare all variants
- [ ] Identify best configurations
- [ ] Prepare visualization materials
- [ ] Write preliminary results section

---

## 11. CHECKLIST SUMMARY

### Core Functionality Improvements
- [ ] Geometric-aware loss functions
- [ ] Multi-scale feature fusion head
- [ ] Uncertainty estimation
- [ ] Gaze-aware pre-training
- [ ] Domain adaptation framework

### Training Enhancements
- [ ] Multi-task learning
- [ ] Contrastive learning
- [ ] Curriculum learning
- [ ] Advanced data augmentation

### Evaluation & Analysis
- [ ] Comprehensive metrics
- [ ] Ablation studies
- [ ] State-of-the-art comparisons
- [ ] Failure analysis
- [ ] Visualization tools

### Documentation & Publication
- [ ] Method description
- [ ] Experimental results
- [ ] Ablation studies
- [ ] Comparison tables
- [ ] Visualization figures
- [ ] Code release

---

## 12. CONCLUSION

This checklist provides a comprehensive roadmap for improving UniGaze with **research-focused contributions** suitable for top-tier journal publication. The improvements are organized by priority and expected impact, with clear implementation steps and success metrics.

**Key Focus Areas:**
1. **Methodological Novelty** - New loss functions, architectures, training strategies
2. **Experimental Rigor** - Comprehensive evaluation, ablation studies, comparisons
3. **Theoretical Understanding** - Analysis of why methods work, generalization bounds
4. **Practical Impact** - Real-world deployment, efficiency, robustness

**Next Steps:**
1. Start with high-priority improvements (loss functions, architecture)
2. Conduct thorough ablation studies
3. Build comprehensive evaluation framework
4. Prepare publication-quality results and analysis

---

**Estimated Timeline:** 6-8 months for complete research cycle  
**Expected Outcome:** Top-tier conference/journal publication with significant improvements over baseline

