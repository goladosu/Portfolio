# A/B Testing Analysis: Frequentist vs. Bayesian & the Peeking Problem

## Project Overview

This project analyzes an e-commerce A/B test comparing a redesigned checkout page against an existing page, using both frequentist and Bayesian statistical frameworks.

### Dataset
- **Size**: 290,584 visitors
- **Period**: January 2-24, 2017 (22 days)
- **Split**: Random 50/50 assignment to control/treatment
- **Geography**: US, UK, Canada

## Key Results

- **Frequentist Test**: p = 0.19 (not significant)
- **Bayesian Analysis**: Only 9.6% probability treatment is better
- **Country Segmentation**: No significant differences in any market
- **Peeking Problem**: Naive monitoring inflates false positives by 4.6×

## Recommendation

**Do not launch the new page**

Both statistical frameworks agree there is insufficient evidence that the redesigned page converts better than the existing page.

## Files

- `ab_clean.csv` - Cleaned experiment data (290K rows)
- `README.md` - This file

## Full Analysis

See the [AB_Testing_Analysis notebook](../../AB_Testing_Analysis) for complete technical details, code, and methodology.

## Integration

This project is integrated into the Portfolio Streamlit app (`app.py`) as a dedicated page with interactive visualizations and detailed explanations.
