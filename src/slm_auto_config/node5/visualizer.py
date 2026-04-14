import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from fpdf import FPDF
import os
import logging
from typing import Optional
from .models import EvaluationMetrics, Node5Metadata

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Generates visual reports, including Confusion Matrices and PDF training summaries.
    """
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.eval_dir = f"runs/{run_id}/evaluation"
        os.makedirs(self.eval_dir, exist_ok=True)

    def plot_confusion_matrix(self, metrics: EvaluationMetrics) -> str:
        """
        Generates a Confusion Matrix heatmap.
        """
        try:
            plt.figure(figsize=(12, 10))
            df_cm = pd.DataFrame(
                metrics.confusion_matrix, 
                index=metrics.labels, 
                columns=metrics.labels
            )
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
            plt.ylabel('Actual Label')
            plt.xlabel('Predicted Label')
            plt.title(f'Confusion Matrix: {self.run_id}')
            
            save_path = f"{self.eval_dir}/confusion_matrix.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return save_path
        except Exception as e:
            logger.error(f"Failed to plot confusion matrix: {e}")
            return ""

    def generate_pdf_report(self, metrics: EvaluationMetrics, metadata: Optional[Node5Metadata] = None) -> str:
        """
        Creates a comprehensive PDF report of the training results.
        """
        try:
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font("Helvetica", "B", 20)
            pdf.cell(0, 15, "SLM Auto Config - Training Report", ln=True, align='C')
            pdf.ln(5)
            
            # Run Info
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(40, 10, "Run ID:", 0)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 10, self.run_id, ln=True)
            
            if metadata and metadata.best_trial_id is not None:
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(40, 10, "Best Trial:", 0)
                pdf.set_font("Helvetica", "", 12)
                pdf.cell(0, 10, f"Trial #{metadata.best_trial_id}", ln=True)
            
            pdf.ln(5)
            
            # Primary Metrics
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Summary Metrics (Test Set)", ln=True)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(50, 8, f"Accuracy: {metrics.accuracy:.4f}", ln=True)
            pdf.cell(50, 8, f"Macro F1 Score: {metrics.macro_f1:.4f}", ln=True)
            
            pdf.ln(10)
            
            # Confusion Matrix
            cm_path = self.plot_confusion_matrix(metrics)
            if os.path.exists(cm_path):
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, "Confusion Matrix", ln=True)
                pdf.image(cm_path, x=10, y=90, w=180)
            
            report_path = f"{self.eval_dir}/report.pdf"
            pdf.output(report_path)
            logger.info(f"Generated PDF report: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            return ""
