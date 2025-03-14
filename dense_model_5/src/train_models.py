import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time
import sys

# Import model definitions
from models import XYErrorNet, ZErrorNet, train_model

def load_data(joint_angles_path, joint_torques_path, actual_xyz_path, nominal_xyz_path):
    """
    Load and prepare the data for training
    
    Args:
        joint_angles_path: Path to joint angles CSV
        joint_torques_path: Path to joint torques CSV
        actual_xyz_path: Path to actual robot positions CSV
        nominal_xyz_path: Path to nominal robot positions CSV
        
    Returns:
        joint_angles: Array of joint angles
        joint_torques: Array of joint torques
        delta_xyz: Array of position errors (nominal - actual)
    """
    # Load joint angles
    joint_angles_df = pd.read_csv(joint_angles_path)
    joint_angles = joint_angles_df[['joint_angle_1', 'joint_angle_2', 'joint_angle_3', 'joint_angle_4', 'joint_angle_5', 'joint_angle_6']].to_numpy(dtype=np.float64)
    
    # Load joint torques
    joint_torques_df = pd.read_csv(joint_torques_path)
    joint_torques = joint_torques_df[['joint_torque_1', 'joint_torque_2', 'joint_torque_3', 'joint_torque_4', 'joint_torque_5', 'joint_torque_6']].to_numpy(dtype=np.float64)
    
    # Load actual and nominal positions
    actual_xyz_df = pd.read_csv(actual_xyz_path)
    actual_xyz = actual_xyz_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    
    nominal_xyz_df = pd.read_csv(nominal_xyz_path)
    nominal_xyz = nominal_xyz_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    
    # Calculate delta (error)
    delta_xyz = nominal_xyz - actual_xyz
    
    print(f"Loaded {len(joint_angles)} data points")
    print(f"Joint angles shape: {joint_angles.shape}")
    print(f"Joint torques shape: {joint_torques.shape}")
    print(f"Delta XYZ shape: {delta_xyz.shape}")
    
    # Print data statistics
    print("\nJoint Angles Statistics:")
    for i in range(6):
        print(f"Joint {i+1}: Min = {np.min(joint_angles[:, i]):.4f}, Max = {np.max(joint_angles[:, i]):.4f}, Mean = {np.mean(joint_angles[:, i]):.4f}")
    
    print("\nJoint Torques Statistics:")
    for i in range(6):
        print(f"Torque {i+1}: Min = {np.min(joint_torques[:, i]):.4f}, Max = {np.max(joint_torques[:, i]):.4f}, Mean = {np.mean(joint_torques[:, i]):.4f}")
    
    print("\nFK Error Statistics (meters):")
    print(f"Delta X: Min = {np.min(delta_xyz[:, 0]):.8f}, Max = {np.max(delta_xyz[:, 0]):.8f}, Mean = {np.mean(delta_xyz[:, 0]):.8f}")
    print(f"Delta Y: Min = {np.min(delta_xyz[:, 1]):.8f}, Max = {np.max(delta_xyz[:, 1]):.8f}, Mean = {np.mean(delta_xyz[:, 1]):.8f}")
    print(f"Delta Z: Min = {np.min(delta_xyz[:, 2]):.8f}, Max = {np.max(delta_xyz[:, 2]):.8f}, Mean = {np.mean(delta_xyz[:, 2]):.8f}")
    
    return joint_angles, joint_torques, delta_xyz

def prepare_data_xy(joint_angles, delta_xy, test_size=0.2, random_state=42):
    """
    Prepare data for XY error prediction model
    
    Args:
        joint_angles: Array of joint angles
        delta_xy: Array of XY position errors
        test_size: Proportion for test split
        random_state: Random seed
        
    Returns:
        Train/test data and scalers
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        joint_angles, delta_xy, test_size=test_size, random_state=random_state
    )
    
    # Scale the joint angles using Min-Max scaling
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale the outputs
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_X, scaler_y

def prepare_data_z(joint_angles, joint_torques, delta_z, test_size=0.2, random_state=42):
    """
    Prepare data for Z error prediction model
    
    Args:
        joint_angles: Array of joint angles
        joint_torques: Array of joint torques
        delta_z: Array of Z position errors
        test_size: Proportion for test split
        random_state: Random seed
        
    Returns:
        Train/test data and scalers
    """
    # Combine joint angles and torques
    joint_data = np.hstack((joint_angles, joint_torques))
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        joint_data, delta_z, test_size=test_size, random_state=random_state
    )
    
    # Scale the inputs using Min-Max scaling
    scaler_X = MinMaxScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Scale the outputs
    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, scaler_X, scaler_y

def visualize_model_predictions(model_name, y_true, y_pred, component_names, save_path):
    """
    Visualize the predicted vs actual values for a model
    
    Args:
        model_name: Name of the model
        y_true: Actual values
        y_pred: Predicted values
        component_names: Names of the components (e.g., ['δx', 'δy'] or ['δz'])
        save_path: Path to save the visualization
    """
    colors = ['b', 'g', 'r'][:len(component_names)]
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of predicted vs actual for each component
    for i, (name, color) in enumerate(zip(component_names, colors)):
        plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, color=color, label=name)
    
    # Add diagonal line for perfect predictions
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    
    plt.xlabel('Actual Error (m)')
    plt.ylabel('Predicted Error (m)')
    plt.title(f'{model_name}: Predicted vs Actual Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Prediction visualization saved to {save_path}")
    

def visualize_combined_predictions(xy_true, xy_pred, z_true, z_pred, save_path='plots/predictions_plot.png'):
    """
    Create a standalone visualization of predictions for all components (X, Y, Z)
    
    Args:
        xy_true: Actual XY error values
        xy_pred: Predicted XY error values
        z_true: Actual Z error values
        z_pred: Predicted Z error values
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(8, 6))

    # Define component names, markers, and grayscale colors
    component_names = ['δx', 'δy', 'δz']
    markers = ['o', 's', '^']  # Circle, Square, Triangle
    colors = ['b', 'g', 'r']  # Grayscale for differentiation
    
    # Plot XY components with different markers
    for i in range(2):
        plt.scatter(
            xy_true[:, i], xy_pred[:, i], alpha=0.6, 
            color=colors[i], marker=markers[i], edgecolors='k', label=component_names[i]
        )
    
    # Plot Z component
    z_true_reshaped = z_true.reshape(-1) if z_true.ndim > 1 and z_true.shape[1] == 1 else z_true
    z_pred_reshaped = z_pred.reshape(-1) if z_pred.ndim > 1 and z_pred.shape[1] == 1 else z_pred
    plt.scatter(
        z_true_reshaped, z_pred_reshaped, alpha=0.6, 
        color=colors[2], marker=markers[2], edgecolors='k', label=component_names[2]
    )
    
    # Add diagonal line for perfect predictions with a dash-dot pattern
    all_values = np.concatenate([xy_true.flatten(), xy_pred.flatten(), 
                                 z_true_reshaped.flatten(), z_pred_reshaped.flatten()])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='dashdot', color='black', label="Perfect Prediction")

    # Labels, title, and grid
    plt.xlabel('Actual Error (m)')
    plt.ylabel('Predicted Error (m)')
    plt.title('Predicted vs Actual Error')
    plt.legend()
    plt.grid(True, linestyle='dotted')
    
    # Save and close the figure
    plt.savefig(save_path)
    plt.close()
    
    print(f"Prediction visualization saved to {save_path}")



def visualize_training_history(xy_history, z_history, save_path='plots/training_history.png'):
    """
    Visualize training history for both models
    
    Args:
        xy_history: Training history for XY model
        z_history: Training history for Z model
        save_path: Path to save the visualization
    """
    plt.figure(figsize=(15, 10))
    
    # Plot XY model training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(xy_history['train_loss'], label='Training Loss')
    plt.plot(xy_history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('XY Model Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot XY model learning rate
    plt.subplot(2, 2, 2)
    plt.plot(xy_history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('XY Model Learning Rate Schedule')
    plt.grid(True)
    
    # Plot Z model training and validation loss
    plt.subplot(2, 2, 3)
    plt.plot(z_history['train_loss'], label='Training Loss')
    plt.plot(z_history['val_loss'], label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Z Model Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Z model learning rate
    plt.subplot(2, 2, 4)
    plt.plot(z_history['learning_rates'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Z Model Learning Rate Schedule')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Training history visualization saved to {save_path}")

def evaluate_model(model, X_test, y_test, scaler_y, device, model_name="Model"):
    """
    Evaluate a trained model on the test set
    
    Args:
        model: Trained model
        X_test: Test inputs
        y_test: Test targets
        scaler_y: Scaler for targets
        device: Device to run on
        model_name: Name for reporting
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_test).cpu().numpy()
    
    # Handle dimensionality for inverse transform
    if y_pred_scaled.shape[1] != scaler_y.scale_.shape[0]:
        # Create a dummy array with correct shape for inverse transform
        dummy_pred = np.zeros((len(y_pred_scaled), scaler_y.scale_.shape[0]))
        dummy_pred[:, :y_pred_scaled.shape[1]] = y_pred_scaled
        y_pred = scaler_y.inverse_transform(dummy_pred)[:, :y_pred_scaled.shape[1]]
    else:
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        
    y_true = y_test.cpu().numpy()
    if y_true.shape[1] != scaler_y.scale_.shape[0]:
        dummy_true = np.zeros((len(y_true), scaler_y.scale_.shape[0]))
        dummy_true[:, :y_true.shape[1]] = y_true
        y_true = scaler_y.inverse_transform(dummy_true)[:, :y_true.shape[1]]
    else:
        y_true = scaler_y.inverse_transform(y_true)
    
    # Calculate MSE, RMSE, MAE
    mse = np.mean((y_pred - y_true) ** 2, axis=0)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_pred - y_true), axis=0)
    
    # Calculate R²
    ss_total = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    ss_residual = np.sum((y_true - y_pred) ** 2, axis=0)
    r2 = 1 - (ss_residual / ss_total)
    
    # Print results
    print(f"\n{model_name} Test Set Evaluation:")
    
    if y_pred.shape[1] == 2:
        print(f"RMSE: δx={rmse[0]:.8f}, δy={rmse[1]:.8f} meters")
        print(f"MAE: δx={mae[0]:.8f}, δy={mae[1]:.8f} meters")
        print(f"R²: δx={r2[0]:.4f}, δy={r2[1]:.4f}")
    else:
        print(f"RMSE: δz={rmse[0]:.8f} meters")
        print(f"MAE: δz={mae[0]:.8f} meters")
        print(f"R²: δz={r2[0]:.4f}")
    
    # Return metrics dictionary
    if y_pred.shape[1] == 2:
        metrics = {
            'rmse': {'x': rmse[0], 'y': rmse[1]},
            'mae': {'x': mae[0], 'y': mae[1]},
            'r2': {'x': r2[0], 'y': r2[1]},
        }
    else:
        metrics = {
            'rmse': {'z': rmse[0]},
            'mae': {'z': mae[0]},
            'r2': {'z': r2[0]},
        }
    
    return metrics, y_true, y_pred

def main():
    original_stdout = sys.stdout
    with open("logs/train_models_log.txt", "w") as f:
        sys.stdout = f
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Set device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Set data paths (update these to your actual paths)
        data_dir = 'data/lara10/data/'
        joint_angles_path = os.path.join(data_dir, "900_joint_angles.csv")
        joint_torques_path = os.path.join(data_dir, "900_joint_torques.csv")
        actual_xyz_path = os.path.join(data_dir, "900_actual_xyz.csv")
        nominal_xyz_path = os.path.join(data_dir, "900_nominal_xyz.csv")
        
        try:
            # Load data
            joint_angles, joint_torques, delta_xyz = load_data(
                joint_angles_path, joint_torques_path, actual_xyz_path, nominal_xyz_path
            )
            
            # Extract XY and Z components
            delta_xy = delta_xyz[:, :2]  # First two columns (x, y)
            delta_z = delta_xyz[:, 2:3]  # Third column (z)
            
            # Prepare data for XY model (uses only joint angles)
            X_train_xy, X_test_xy, y_train_xy, y_test_xy, scaler_X_xy, scaler_y_xy = prepare_data_xy(
                joint_angles, delta_xy, test_size=0.2, random_state=42
            )
            
            # Prepare data for Z model (uses joint angles + torques)
            X_train_z, X_test_z, y_train_z, y_test_z, scaler_X_z, scaler_y_z = prepare_data_z(
                joint_angles, joint_torques, delta_z, test_size=0.2, random_state=42
            )
            
            print(f"\nXY Model - Training set size: {len(X_train_xy)}, Test set size: {len(X_test_xy)}")
            print(f"Z Model - Training set size: {len(X_train_z)}, Test set size: {len(X_test_z)}")
            
            # Create models
            xy_model = XYErrorNet(dropout_rate=0.25)
            z_model = ZErrorNet(dropout_rate=0.25)
            
            print(f"\nXY Model architecture:\n{xy_model}")
            print(f"\nZ Model architecture:\n{z_model}")
            
            # Count parameters
            xy_params = sum(p.numel() for p in xy_model.parameters())
            z_params = sum(p.numel() for p in z_model.parameters())
            print(f"\nXY Model parameters: {xy_params:,}")
            print(f"Z Model parameters: {z_params:,}")
            
            # Train XY model
            print("\n=== Training XY Model ===")
            trained_xy_model, xy_history = train_model(
                xy_model, X_train_xy, y_train_xy, X_test_xy, y_test_xy, scaler_y_xy,
                batch_size=64, epochs=150, lr=0.001, patience=15, device=device,
                model_name="XY Model"
            )
            
            # Train Z model
            print("\n=== Training Z Model ===")
            trained_z_model, z_history = train_model(
                z_model, X_train_z, y_train_z, X_test_z, y_test_z, scaler_y_z,
                batch_size=64, epochs=150, lr=0.001, patience=15, device=device,
                model_name="Z Model"
            )
            
            # Visualize training history
            visualize_training_history(xy_history, z_history)
            
            # Evaluate models (now returns true and predicted values)
            xy_metrics, xy_true, xy_pred = evaluate_model(trained_xy_model, X_test_xy, y_test_xy, scaler_y_xy, device, "XY Model")
            z_metrics, z_true, z_pred = evaluate_model(trained_z_model, X_test_z, y_test_z, scaler_y_z, device, "Z Model")
            
            # Visualize individual model predictions
            visualize_model_predictions("XY Model", xy_true, xy_pred, ['δx', 'δy'], 'plots/xy_predictions.png')
            visualize_model_predictions("Z Model", z_true, z_pred, ['δz'], 'plots/z_predictions.png')
            
            # Visualize combined predictions
            visualize_combined_predictions(xy_true, xy_pred, z_true, z_pred, 'plots/combined_predictions.png')
            
            # Save models
            torch.save({
                'model_state_dict': trained_xy_model.state_dict(),
                'scaler_X': scaler_X_xy,
                'scaler_y': scaler_y_xy,
                'metrics': xy_metrics
            }, 'models/xy_error_model.pth')
            
            torch.save({
                'model_state_dict': trained_z_model.state_dict(),
                'scaler_X': scaler_X_z,
                'scaler_y': scaler_y_z,
                'metrics': z_metrics
            }, 'models/z_error_model.pth')
            
            print("\nModels saved to 'models/xy_error_model.pth' and 'models/z_error_model.pth'")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Please check your data paths and file formats.")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()