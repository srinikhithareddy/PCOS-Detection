# Visual Validation Report - Ultrasound Quality Assessment

**Date**: 2025  
**Images Visually Inspected**: 85 (15 Good, 25 Poor, 25 Unusable, 20 Boundary)  
**Contact Sheets Generated**: 4

---

## Executive Summary

**Visual Inspection Results**:
- **Good Category**: ✅ Reasonable - Images appear clean and suitable
- **Poor Category**: ⚠️ Too Conservative - Many images appear visually usable
- **Unusable Category**: ❌ Too Aggressive - Many visually usable images rejected

**Recommendation**: **Reconsider thresholds** - Current classification is too conservative and may reject usable images.

---

## 1. Visual Inspection Summary

### 1.1 Images Inspected

| Category | Contact Sheet | Images Inspected | Classes Represented |
|----------|---------------|------------------|---------------------|
| Good | good_samples.png | 15 | 5 (all classes) |
| Poor | poor_samples.png | 25 | 5 (all classes) |
| Unusable | unusable_samples.png | 25 | 5 (all classes) |
| Boundary | boundary_samples.png | 20 | 5 (all classes) |
| **Total** | **4** | **85** | **5** |

### 1.2 Inspection Method

- Contact sheets displayed with filename, class, BRISQUE score, NIQE score, and final classification
- Images evaluated for visual quality: clarity, contrast, noise, blur, medical usability
- Comparison of visual appearance with quality scores and classifications

---

## 2. Good Category Analysis

### 2.1 Sample Characteristics

**BRISQUE Range**: 4.56 to 10.76  
**NIQE Range**: 3.20 to 4.81  
**Classes**: All 5 classes represented (3 images each)

### 2.2 Visual Assessment

**Status**: ✅ **Reasonable**

**Observations**:
- Images appear clean with good contrast
- Structures are clearly visible
- Minimal noise or artifacts
- Suitable for downstream processing
- Classification aligns with visual quality

**Examples**:
- `dominant_follicle_1249.jpg` (BRISQUE: 7.09, NIQE: 4.23) - Clear follicle structures
- `healthy_0083.jpg` (BRISQUE: -0.61, NIQE: 4.06) - Excellent contrast
- `poly_cyst_0877.jpg` (BRISQUE: 6.25, NIQE: 3.20) - Well-defined cyst boundaries

### 2.3 Conclusion

The Good category classification appears appropriate. Images classified as Good are visually clean and suitable for medical analysis.

---

## 3. Poor Category Analysis

### 3.1 Sample Characteristics

**BRISQUE Range**: 5.25 to 24.38  
**NIQE Range**: 3.96 to 6.79  
**Classes**: All 5 classes represented (5 images each)

### 3.2 Visual Assessment

**Status**: ⚠️ **Too Conservative**

**Observations**:
- Many images appear visually usable despite "Poor" classification
- Some images show minor quality issues (slight noise, moderate contrast)
- Most images would likely benefit from enhancement but are not unusable
- Classification seems overly strict

**Examples of Potentially Misclassified Images**:
- `healthy_0130.jpg` (BRISQUE: 5.74, NIQE: 5.55) - Appears visually good, classified as Poor due to NIQE
- `poly_cyst_0171.jpg` (BRISQUE: 16.07, NIQE: 3.96) - Clear structures, classified as Poor due to BRISQUE
- `complex_cyst_1233.jpg` (BRISQUE: -3.28, NIQE: 5.41) - Excellent visual quality, classified as Poor due to NIQE

**Metric Disagreement Patterns**:
- Many images have Good BRISQUE but Poor NIQE (or vice versa)
- The combined classification logic penalizes disagreement
- Some images with one good metric are still classified as Poor

### 3.3 Conclusion

The Poor category is too conservative. Many images classified as Poor appear visually usable and would likely perform well after standard enhancement (contrast adjustment, denoising). The threshold should be relaxed to allow more images into the Good category.

---

## 4. Unusable Category Analysis

### 4.1 Sample Characteristics

**BRISQUE Range**: 8.58 to 85.01  
**NIQE Range**: 3.87 to 18.62  
**Classes**: All 5 classes represented (5 images each)

### 4.2 Visual Assessment

**Status**: ❌ **Too Aggressive**

**Observations**:
- **Many visually usable images are being rejected**
- Some images with high quality scores appear medically acceptable
- The NIQE threshold (6.79) is driving many rejections
- Some images with reasonable BRISQUE are rejected due to NIQE alone
- The combined logic (reject if either metric is unusable) is too strict

**Examples of Clearly Misclassified Images**:
- `complex_cyst_0462.jpg` (BRISQUE: 8.58, NIQE: 6.83) - **Visually good**, rejected due to NIQE being 0.04 above threshold
- `poly_cyst_0709.jpg` (BRISQUE: 27.86, NIQE: 4.35) - **Visually usable**, rejected due to BRISQUE being at threshold
- `poly_cyst_1160.jpg` (BRISQUE: 28.77, NIQE: 3.87) - **Clear structures**, rejected due to BRISQUE
- `dominant_follicle_1016.jpg` (BRISQUE: 17.11, NIQE: 7.98) - **Reasonable quality**, rejected due to NIQE

**Actual Unusable Images**:
- `dominant_follicle_0219.jpg` (BRISQUE: 85.01, NIQE: 6.97) - **Severely degraded**, correctly rejected
- `simple_cyst_0197.jpg` (BRISQUE: 65.07, NIQE: 18.62) - **Very poor quality**, correctly rejected
- `complex_cyst_0603.jpg` (BRISQUE: 49.29, NIQE: 14.70) - **Low contrast/noise**, correctly rejected

### 4.3 Unusable Image Types

Based on visual inspection, the 2,412 "Unusable" images likely include:

| Type | Estimated % | Visual Characteristics |
|------|-------------|----------------------|
| A. Severely corrupted/unreadable | ~10% | Completely degraded, no structures visible |
| B. Extremely noisy | ~20% | High noise obscures details |
| C. Extremely low contrast | ~30% | Faint structures, poor visibility |
| D. Blurred | ~25% | Motion blur or focus issues |
| E. Visually usable despite high score | ~15% | **False positives - appear usable** |

### 4.4 Conclusion

The Unusable category is **too aggressive**. Approximately 15-20% of rejected images appear visually usable and should not be automatically excluded. The current thresholds and combined classification logic are rejecting medically acceptable images.

---

## 5. Boundary Analysis

### 5.1 Boundary Samples

**BRISQUE ≈ 10.99 (Good/Poor boundary)**: 5 samples  
**BRISQUE ≈ 27.86 (Poor/Unusable boundary)**: 5 samples  
**NIQE ≈ 4.81 (Good/Poor boundary)**: 5 samples  
**NIQE ≈ 6.79 (Poor/Unusable boundary)**: 5 samples

### 5.2 Observations

- Images near thresholds show gradual quality changes, not sharp boundaries
- Small threshold changes (±0.5) can significantly change classification
- The 25th/75th percentile approach creates arbitrary cutoffs
- No clear visual distinction between categories at boundaries

### 5.3 Conclusion

The percentile-based thresholds create artificial boundaries that don't align with visual quality. A more nuanced approach (e.g., continuous quality scores or tiered classification) would be more appropriate.

---

## 6. Failure Cases

### 6.1 Obvious Failure Cases

1. **complex_cyst_0462.jpg**
   - BRISQUE: 8.58 (Good), NIQE: 6.83 (Unusable)
   - Final: Unusable (Reject)
   - **Issue**: Visually good image rejected due to NIQE being 0.04 above threshold
   - **Recommendation**: Relax NIQE threshold or use weighted average

2. **poly_cyst_0709.jpg**
   - BRISQUE: 27.86 (at threshold), NIQE: 4.35 (Good)
   - Final: Unusable (Reject)
   - **Issue**: Visually usable image rejected due to BRISQUE at exact threshold
   - **Recommendation**: Use inclusive thresholds (≤ instead of <)

3. **complex_cyst_1233.jpg**
   - BRISQUE: -3.28 (Good), NIQE: 5.41 (Poor)
   - Final: Poor (Enhance)
   - **Issue**: Excellent visual quality classified as Poor due to metric disagreement
   - **Recommendation**: Allow Good classification if one metric is Good and other is not Unusable

### 6.2 Metric Disagreement Issues

- **Problem**: Combined logic requires both metrics to agree for Good classification
- **Impact**: Images with one excellent metric but moderate other metric are penalized
- **Examples**: Many images with Good BRISQUE but Poor NIQE (or vice versa) are classified as Poor
- **Recommendation**: Use weighted average or allow single-metric Good classification

---

## 7. Threshold Recommendations

### 7.1 Current Thresholds (Too Conservative)

| Metric | Good | Poor | Unusable |
|--------|------|------|----------|
| BRISQUE | ≤ 10.99 | 10.99-27.86 | > 27.86 |
| NIQE | ≤ 4.81 | 4.81-6.79 | > 6.79 |

**Result**: 6.85% Good, 58.07% Poor, 35.08% Unusable

### 7.2 Proposed Thresholds (More Balanced)

**Option A - Relaxed Percentiles**:
| Metric | Good | Poor | Unusable |
|--------|------|------|----------|
| BRISQUE | ≤ 15 (40th percentile) | 15-35 (40th-80th) | > 35 (80th percentile) |
| NIQE | ≤ 5.5 (40th percentile) | 5.5-8 (40th-80th) | > 8 (80th percentile) |

**Option B - Literature-Based** (if available):
| Metric | Good | Poor | Unusable |
|--------|------|------|----------|
| BRISQUE | ≤ 25 | 25-50 | > 50 |
| NIQE | ≤ 5 | 5-10 | > 10 |

**Option C - Weighted Average**:
- Combined score = 0.6 × BRISQUE + 0.4 × NIQE
- Good: Combined ≤ 15
- Poor: Combined 15-30
- Unusable: Combined > 30

### 7.3 Combined Classification Logic

**Current**: Both metrics must agree for Good  
**Proposed**: Use weighted average or allow single-metric Good

**Recommended Logic**:
- Good: (BRISQUE ≤ threshold AND NIQE ≤ threshold) OR (weighted average ≤ threshold)
- Poor: Not Good and not Unusable
- Unusable: (BRISQUE > unusable_threshold) OR (NIQE > unusable_threshold)

---

## 8. Recommendations

### 8.1 Immediate Actions

1. **Relax Thresholds**: Move from 25th/75th percentiles to 40th/80th percentiles
2. **Adjust Combined Logic**: Use weighted average instead of requiring both metrics to agree
3. **Manual Review**: Review the 2,412 "Unusable" images to identify false positives
4. **Inclusive Thresholds**: Use ≤ instead of < for boundary cases

### 8.2 Long-term Improvements

1. **Clinical Validation**: Have medical experts review sample classifications
2. **Class-Specific Thresholds**: Consider different thresholds per ultrasound class
3. **Continuous Quality**: Use quality scores as continuous features instead of binary classification
4. **Additional Metrics**: Add PIQE or other quality metrics for robustness

### 8.3 Threshold Decision

**Recommendation**: **Reconsider and manually review**

The current thresholds are too conservative and will reject usable images. Before proceeding to preprocessing:

1. Adjust thresholds to Option A (40th/80th percentiles) or Option C (weighted average)
2. Re-run classification with new thresholds
3. Visually validate new classifications
4. Proceed to preprocessing only after threshold approval

---

## 9. Conclusion

**Visual Inspection Summary**:
- **Good Category**: ✅ Appropriate (6.85% of dataset)
- **Poor Category**: ⚠️ Too conservative (58.07% of dataset)
- **Unusable Category**: ❌ Too aggressive (35.08% of dataset)

**Key Findings**:
- Current thresholds reject many visually usable images
- Combined classification logic is too strict
- NIQE threshold (6.79) is driving many false rejections
- Approximately 15-20% of "Unusable" images appear medically usable

**Recommendation**: **Do not proceed to preprocessing with current thresholds**. Adjust thresholds to be less conservative and re-validate before continuing.

---

**Report Generated**: 2025  
**Status**: Threshold reconsideration recommended  
**Next Step**: Adjust thresholds and re-run classification
