# scripts/generate_reports.py
import pandas as pd

def generate_report(predictions, mae):
    report = f"Model Evaluation Report\n\nMean Absolute Error (MAE): {mae}\n"
    report += "Predictions:\n" + str(predictions)
    
    # Save the report to a text file
    with open('reports/evaluation_report.txt', 'w') as f:
        f.write(report)

