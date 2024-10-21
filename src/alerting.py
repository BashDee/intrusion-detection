def alert_system(prediction):
    """Alerts if the prediction corresponds to an attack"""
    if prediction == 1:  # Assuming '1' represents an attack
        print("ALERT: Potential Attack Detected!")
    else:
        print("Normal Traffic")

# Usage Example
if __name__ == '__main__':
    predictions = [0, 1, 1, 0]  # Sample predictions
    for pred in predictions:
        alert_system(pred)
