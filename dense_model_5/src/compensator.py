import torch
import numpy as np
import time
import pandas as pd
import os
import sys

# Import model definitions
from models import XYErrorNet, ZErrorNet

class FKErrorCompensator:
    """
    Class for compensating forward kinematics errors using trained neural networks.
    Uses separate models for XY and Z error prediction.
    """
    def __init__(self, xy_model_path='models/xy_error_model.pth', z_model_path='models/z_error_model.pth', device='cpu'):
        """
        Initialize the compensator by loading the trained models and scalers.
        
        Args:
            xy_model_path: Path to the saved XY model file
            z_model_path: Path to the saved Z model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load XY model (joint angles only)
        self.xy_model = None
        self.xy_scaler_X = None
        self.xy_scaler_y = None
        self.load_xy_model(xy_model_path)
        
        # Load Z model (joint angles + torques)
        self.z_model = None
        self.z_scaler_X = None
        self.z_scaler_y = None
        self.load_z_model(z_model_path)
    
    def load_xy_model(self, model_path):
        """
        Load the trained XY error model and its scalers.
        
        Args:
            model_path: Path to the saved model file
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model and load weights
            self.xy_model = XYErrorNet()
            self.xy_model.load_state_dict(checkpoint['model_state_dict'])
            self.xy_model.to(self.device)
            self.xy_model.eval()
            
            # Load scalers
            self.xy_scaler_X = checkpoint['scaler_X']
            self.xy_scaler_y = checkpoint['scaler_y']
            
            # Load metrics (if available)
            self.xy_metrics = checkpoint.get('metrics', None)
            
            print(f"XY model loaded successfully from {model_path}")
            if self.xy_metrics:
                print(f"XY model metrics - RMSE: δx={self.xy_metrics['rmse']['x']:.8f}, δy={self.xy_metrics['rmse']['y']:.8f}")
        
        except Exception as e:
            print(f"Error loading XY model: {e}")
            raise
    
    def load_z_model(self, model_path):
        """
        Load the trained Z error model and its scalers.
        
        Args:
            model_path: Path to the saved model file
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Create model and load weights
            self.z_model = ZErrorNet()
            self.z_model.load_state_dict(checkpoint['model_state_dict'])
            self.z_model.to(self.device)
            self.z_model.eval()
            
            # Load scalers
            self.z_scaler_X = checkpoint['scaler_X']
            self.z_scaler_y = checkpoint['scaler_y']
            
            # Load metrics (if available)
            self.z_metrics = checkpoint.get('metrics', None)
            
            print(f"Z model loaded successfully from {model_path}")
            if self.z_metrics:
                print(f"Z model metrics - RMSE: δz={self.z_metrics['rmse']['z']:.8f}")
        
        except Exception as e:
            print(f"Error loading Z model: {e}")
            raise
    
    def predict_error(self, joint_angles, joint_torques=None):
        """
        Predict the FK position error using separate models for XY and Z.
        
        Args:
            joint_angles: Array of joint angles (shape: [n_samples, 6] or [6])
            joint_torques: Array of joint torques (shape: [n_samples, 6] or [6])
                           If None, uses zeros for Z prediction
        
        Returns:
            predicted_errors: Predicted position errors (shape: [n_samples, 3] or [3])
        """
        # Handle single sample (convert to 2D array)
        single_sample = False
        if isinstance(joint_angles, list):
            joint_angles = np.array(joint_angles)
        
        if len(joint_angles.shape) == 1:
            joint_angles = joint_angles.reshape(1, -1)
            single_sample = True
        
        # If no torques provided, use zeros
        if joint_torques is None:
            joint_torques = np.zeros_like(joint_angles)
        elif isinstance(joint_torques, list):
            joint_torques = np.array(joint_torques)
        
        if len(joint_torques.shape) == 1:
            joint_torques = joint_torques.reshape(1, -1)
        
        # Predict XY errors (using only joint angles)
        X_angles_scaled = self.xy_scaler_X.transform(joint_angles)
        X_angles_tensor = torch.tensor(X_angles_scaled, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            xy_pred_scaled = self.xy_model(X_angles_tensor).cpu().numpy()
        
        # Inverse transform XY predictions
        xy_pred = self.xy_scaler_y.inverse_transform(xy_pred_scaled)
        
        # Prepare combined data for Z prediction
        X_combined = np.hstack((joint_angles, joint_torques))
        X_combined_scaled = self.z_scaler_X.transform(X_combined)
        X_combined_tensor = torch.tensor(X_combined_scaled, dtype=torch.float32).to(self.device)
        
        # Predict Z error
        with torch.no_grad():
            z_pred_scaled = self.z_model(X_combined_tensor).cpu().numpy()
        
        # Inverse transform Z prediction
        # Create a dummy array with the right shape for inverse transform
        dummy_z = np.zeros((len(z_pred_scaled), self.z_scaler_y.scale_.shape[0]))
        dummy_z[:, 0:z_pred_scaled.shape[1]] = z_pred_scaled
        z_pred = self.z_scaler_y.inverse_transform(dummy_z)[:, 0:z_pred_scaled.shape[1]]
        
        # Combine predictions
        predicted_errors = np.hstack((xy_pred, z_pred))
        
        # Return original shape for single sample
        if single_sample:
            predicted_errors = predicted_errors.flatten()
        
        return predicted_errors
    
    def compensate_position(self, joint_angles, joint_torques, nominal_position):
        """
        Compensate the nominal FK position by subtracting the predicted error.
        
        Args:
            joint_angles: Array of joint angles (shape: [6] or [n_samples, 6])
            joint_torques: Array of joint torques (shape: [6] or [n_samples, 6])
            nominal_position: Nominal position from FK (shape: [3] or [n_samples, 3])
            
        Returns:
            compensated_position: Error-compensated position
        """
        # Predict error
        predicted_errors = self.predict_error(joint_angles, joint_torques)
        
        # Convert to arrays if needed
        if isinstance(nominal_position, list):
            nominal_position = np.array(nominal_position)
        
        # Handle shape matching
        if len(predicted_errors.shape) == 1 and len(nominal_position.shape) == 1:
            # Both are single samples
            return nominal_position - predicted_errors
        elif len(predicted_errors.shape) == 2 and len(nominal_position.shape) == 1:
            # Multiple error predictions but single nominal position
            nominal_position = np.tile(nominal_position, (predicted_errors.shape[0], 1))
            return nominal_position - predicted_errors
        elif len(predicted_errors.shape) == 1 and len(nominal_position.shape) == 2:
            # Single error prediction but multiple nominal positions
            predicted_errors = np.tile(predicted_errors, (nominal_position.shape[0], 1))
            return nominal_position - predicted_errors
        else:
            # Both are batches
            return nominal_position - predicted_errors
    
    def benchmark_performance(self, num_samples=1000):
        """
        Benchmark the prediction speed of the models.
        
        Args:
            num_samples: Number of samples to use for benchmarking
            
        Returns:
            avg_time: Average time per prediction (ms)
        """
        # Generate random joint angles and torques
        joint_angles = np.random.uniform(-np.pi, np.pi, (num_samples, 6))
        joint_torques = np.random.uniform(-10, 10, (num_samples, 6))
        
        # Warm-up
        for _ in range(10):
            _ = self.predict_error(joint_angles[0:1], joint_torques[0:1])
        
        # Benchmark batch prediction
        start_time = time.time()
        _ = self.predict_error(joint_angles, joint_torques)
        batch_time = (time.time() - start_time) * 1000  # ms
        
        # Benchmark single predictions
        single_times = []
        for i in range(100):  # Test with 100 samples for single prediction
            start_time = time.time()
            _ = self.predict_error(joint_angles[i], joint_torques[i])
            single_times.append((time.time() - start_time) * 1000)  # ms
        
        avg_single_time = np.mean(single_times)
        
        print(f"Batch prediction time for {num_samples} samples: {batch_time:.2f} ms")
        print(f"Average single prediction time: {avg_single_time:.2f} ms")
        print(f"Predictions per second (single): {1000/avg_single_time:.1f}")
        
        return avg_single_time


# Example usage
if __name__ == "__main__":
    original_stdout = sys.stdout
    with open("logs/compensator_log.txt", "w") as f:
        sys.stdout = f
    
        # Create compensator
        compensator = FKErrorCompensator(
            xy_model_path='models/xy_error_model.pth',
            z_model_path='models/z_error_model.pth'
        )
        
        # Benchmark performance
        avg_time = compensator.benchmark_performance(1000)
        
        # Example: Compensate a position

        joint_angles = np.array([-2.74881040458028, -0.986414329324354, 0.940812160018306, 0.552588811814705, -1.63338568014575, -2.63111156625165])
        joint_torques = np.array([-6.166765, 67.7470464285714, 9.61920817142857, 1.92507565714286, 3.87989328571429, -1.80596585714286])
        nominal_position = np.array([0.941128004226422, -1.85795226184815, 0.196124527040959])
        actual_positionn = np.array([0.94451502576358, -1.85499312097213, 0.195208646376031])
        
        # Predict error
        original_error = actual_positionn - nominal_position
        predicted_error = compensator.predict_error(joint_angles, joint_torques)
        print(f"\nJoint angles: {joint_angles}")
        print(f"Joint torques: {joint_torques}")
        print(f"Predicted error: {predicted_error} meters")
        print(f"Original error: {original_error} meters")
        
        # Compensate position
        compensated_position = compensator.compensate_position(joint_angles, joint_torques, nominal_position)
        print(f"Nominal position: {nominal_position} meters")
        print(f"Compensated position: {compensated_position} meters")
        
        # Test with multiple configurations
        data_dir = 'data/lara10/data/'
        joint_angles_path = os.path.join(data_dir, "70_joint_angles.csv")
        joint_torques_path = os.path.join(data_dir, "70_joint_torques.csv")
        actual_xyz_path = os.path.join(data_dir, "70_actual_xyz.csv")
        nominal_xyz_path = os.path.join(data_dir, "70_nominal_xyz.csv")
        
        joint_angles_df = pd.read_csv(joint_angles_path)
        joint_angles_batch = joint_angles_df[['joint_angle_1', 'joint_angle_2', 'joint_angle_3', 'joint_angle_4', 'joint_angle_5', 'joint_angle_6']].to_numpy(dtype=np.float64)
        
        joint_torques_df = pd.read_csv(joint_torques_path)
        joint_torques_batch = joint_torques_df[['joint_torque_1', 'joint_torque_2', 'joint_torque_3', 'joint_torque_4', 'joint_torque_5', 'joint_torque_6']].to_numpy(dtype=np.float64)
        
        nominal_xyz_df = pd.read_csv(nominal_xyz_path)
        nominal_position_batch = nominal_xyz_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)
        
        compensated_positions = compensator.compensate_position(
            joint_angles_batch, joint_torques_batch, nominal_position_batch
        )
        
        
        print("\nBatch compensation example:")
        for i in range(5):
            print(f"Sample {i+1}:")
            print(f"  Nominal: [{nominal_position_batch[i][0]:.6f}, {nominal_position_batch[i][1]:.6f}, {nominal_position_batch[i][2]:.6f}]")
            print(f"  Compensated: [{compensated_positions[i][0]:.6f}, {compensated_positions[i][1]:.6f}, {compensated_positions[i][2]:.6f}]")
            print(f"  Delta: [{compensated_positions[i][0]-nominal_position_batch[i][0]:.6f}, "
                f"{compensated_positions[i][1]-nominal_position_batch[i][1]:.6f}, "
                f"{compensated_positions[i][2]-nominal_position_batch[i][2]:.6f}]")
        