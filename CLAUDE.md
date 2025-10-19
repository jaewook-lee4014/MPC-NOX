# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a NOx (Nitrogen Oxides) optimization and control project that analyzes industrial boiler emissions data using machine learning techniques. The project focuses on predicting and optimizing NOx emissions from time-series sensor data collected from industrial boiler operations.

## Repository Structure

```
OptCont-NOX/
├── 0.Data/                           # Time-series sensor data directory
│   ├── label.txt                     # Column labels for sensor measurements (56 features)
│   └── ocr_results_YYYY-MM-DD.csv    # Daily sensor readings with timestamps
└── NOx_feature_overview.ipynb        # Main analysis notebook
```

## Data Architecture

### Data Format
- **Sensor Data**: CSV files with 57 columns (timestamp + 56 sensor measurements)
- **Frequency**: ~12-second intervals (approximately 7,000+ records per day)
- **Target Variable**: Column 43 "NOx (ppm)" - primary emission metric to predict/optimize
- **Secondary Target**: Column 40 "NOx" - alternative NOx measurement

### Key Sensor Categories (from label.txt)
- **Steam System**: Main Steam Flow, Pressure, Drum Level/Pressure
- **Air/Fuel Control**: Air Flow, Air Control Valve Opening, Fuel Gas Pressure
- **Temperature Monitoring**: Gas Temp In/Out, Bunker Air Temp, SAH Temp Out
- **Emissions**: NOx, CO, SOx, O2, HCL, DUST
- **Urea Injection**: UREA FLOW (L/H and T/H) - SCR system for NOx reduction
- **Operational Parameters**: Damper positions, pressures, loads

### Data Processing Pipeline
1. **Data Loading**: CSV files loaded with timestamp + label mapping
2. **Preprocessing**: 
   - Remove zero-variance and low-uniqueness columns
   - Fill missing values with mean
   - MinMax normalization (0-1 scale)
3. **Feature Selection**: Correlation analysis with NOx target (threshold ≥ 0.2)
4. **Model Training**: 80/20 time-based split for temporal validation

## Working with This Codebase

### Data Analysis Workflow
1. **Load Data**: Use `NOx_feature_overview.ipynb` as the primary analysis notebook
2. **Feature Analysis**: The notebook identifies the top correlated features with NOx emissions:
   - Main Steam Flow (0.449 correlation)
   - Air & Fuel Gas Pressure (0.422)
   - Air Control Valve Opening (0.419)
   - Lower Heating Value (0.417)
3. **Model Development**: MLPRegressor with (64, 32, 16) hidden layers is the baseline model

### Key Implementation Notes
- **Target Variable**: Use column 43 "NOx (ppm)" as the primary optimization target
- **Feature Engineering**: High correlation features (>0.2) are pre-identified in the notebook
- **Temporal Aspects**: Data has strong time-series characteristics - use chronological train/test splits
- **Korean Language**: Some visualization uses Korean fonts (AppleSDGothicNeo.ttc on macOS)

### Data Access Patterns
- Individual day analysis: Filter by date from timestamp column
- Multi-day analysis: Concatenate multiple CSV files from 0.Data/ directory
- Missing data handling: Some sensors have occasional missing values (e.g., P3, CO_TMS)

## Development Environment

This is a Jupyter notebook-based project with no specific build/test commands. The main development workflow involves:

1. **Run Jupyter**: Start Jupyter notebook server to access `NOx_feature_overview.ipynb`
2. **Data Analysis**: Execute cells sequentially for data exploration and model development
3. **Model Iteration**: Modify feature selection and model parameters based on correlation analysis

## Dependencies

The project uses standard data science libraries:
- pandas, numpy (data manipulation)
- scikit-learn (machine learning, preprocessing)
- matplotlib (visualization with Korean font support)

Note: No package.json, requirements.txt, or other dependency management files are present in the repository.