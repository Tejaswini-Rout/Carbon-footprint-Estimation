from model import train_model
from predictor import make_predictions, predict_for_date

if __name__ == "__main__":
    train_model()
    make_predictions()
    predict_for_date("2025-06-15")
    print("Execution complete!")
