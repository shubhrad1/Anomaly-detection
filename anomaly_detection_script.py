# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Function to simulate a data stream with pattern, seasonality, and noise
def generate_data_stream(size=1000, freq=0.01, seasonality_freq=0.05, noise_level=1):
    t = np.arange(size)
    signal = np.sin(2 * np.pi * freq * t)
    seasonality = 0.5 * np.sin(2 * np.pi * seasonality_freq * t)
    noise = noise_level * np.random.randn(size)
    return signal + seasonality + noise

# Function for anomaly detection using Isolation Forest
def detect_anomalies(data, model,contamination):
    anomalies=model.predict(data.reshape(-1,1))
    return anomalies

# Function for real-time visualization
def visualize_data_stream(data, anomalies, threshold=0):
    plt.plot(data, label='Data Stream')
    plt.scatter(np.where(anomalies < threshold), data[anomalies < threshold], color='red', label='Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Final Data Stream Visualization with Anomalies')
    plt.legend()
    plt.show()

# Main script
if __name__ == "__main__":
    # Simulate data stream
    data_stream = generate_data_stream()

    # Set up parameters
    contamination = 0.03  #contamination value set after multiple experimentations

    initial_training_size = int(0.6 * len(data_stream))  #Training set size. Uses 60% of data stream for training.

    #Model definition
    model=IsolationForest(contamination=contamination)

    # Training the model
    model.fit(data_stream[:initial_training_size].reshape(-1,1))

    accumulated_data=[]

    # Real-time anomaly detection
    for i in range(len(data_stream)):
        current_data_point = data_stream[i]

        # Accumulate data
        accumulated_data.append(current_data_point)


    # Final anomaly detection
    anomalies = detect_anomalies(np.array(accumulated_data), model, contamination)

    # Final visualization
    visualize_data_stream(np.array(accumulated_data), anomalies, threshold=0)