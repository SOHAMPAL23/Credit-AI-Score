"""
Quick Demo Script for Optimized Loan Approval Model
Run this to quickly test the enhanced model performance
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def run_notebook_demo():
    """Run the Jupyter notebook demonstration"""
    print("\n🔬 Starting Jupyter Notebook Demo...")
    print("This will open the optimized loan approval model notebook")
    print("Please run all cells to see the complete workflow")
    
    try:
        subprocess.run(["jupyter", "notebook", "loan_approval_model_optimized.ipynb"])
    except FileNotFoundError:
        print("❌ Jupyter not found. Please install it with: pip install jupyter")
        return False
    except Exception as e:
        print(f"❌ Error opening notebook: {e}")
        return False
    return True

def run_predictor_demo():
    """Run the predictor demonstration"""
    print("\n🤖 Running Predictor Demo...")
    
    try:
        # Import and run the predictor
        from loan_predictor import demo_predictions
        demo_predictions()
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Error running predictor: {e}")
        return False

def main():
    print("🏦 LOAN APPROVAL MODEL - OPTIMIZED VERSION")
    print("=" * 50)
    print("This enhanced version includes:")
    print("• Multi-algorithm comparison (RF, XGBoost, GBM)")
    print("• Advanced hyperparameter tuning")
    print("• 14 engineered features")
    print("• Comprehensive performance evaluation")
    print("• Cross-validation and model interpretability")
    print()
    
    # Check if requirements are met
    choice = input("Choose an option:\n1. Install requirements and run full demo\n2. Run predictor demo only\n3. Exit\nEnter choice (1-3): ")
    
    if choice == "1":
        if install_requirements():
            print("\n🎉 Setup complete! Now you can:")
            print("1. Run the Jupyter notebook for full analysis")
            print("2. Use the predictor script for quick predictions")
            print("3. Check OPTIMIZED_README.md for detailed documentation")
            
    elif choice == "2":
        run_predictor_demo()
        
    elif choice == "3":
        print("👋 Goodbye!")
        return
        
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()