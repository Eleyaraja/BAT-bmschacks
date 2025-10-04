import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-normal {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .prediction-pneumonia {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .confidence-box {
        background-color: #e2e3e5;
        border-left: 5px solid #6c757d;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model Architecture Classes
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        return out * self.spatial_attention(out)

class PneumoniaClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b3', num_classes=2, dropout_rate=0.3):
        super(PneumoniaClassifier, self).__init__()
        
        try:
            import timm
        except ImportError:
            st.error("Please install timm: pip install timm")
            st.stop()
        
        if 'efficientnet' in model_name:
            self.backbone = timm.create_model(model_name, pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        
        self.attention = CBAM(num_features)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.attention(features)
        return self.classifier(features)

# Transform function
def get_transforms():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# CORRECTED: Load model function for your specific structure
@st.cache_resource
def load_ensemble_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Handle PyTorch loading issues
        torch.serialization.add_safe_globals(["numpy.core.multiarray.scalar"])
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        st.info(f"🔍 Loaded checkpoint with keys: {list(checkpoint.keys())}")
        
        models = []
        
        # Method 1: Try ensemble_state_dict (single ensemble model)
        if 'ensemble_state_dict' in checkpoint:
            st.success("🎯 Found ensemble_state_dict - loading single ensemble model")
            
            model = PneumoniaClassifier(
                model_name='efficientnet_b3',
                num_classes=2,
                dropout_rate=0.0
            ).to(device)
            
            model.load_state_dict(checkpoint['ensemble_state_dict'])
            model.eval()
            models.append(model)
            
        # Method 2: Try individual_models (multiple models)
        elif 'individual_models' in checkpoint:
            individual_models = checkpoint['individual_models']
            st.success(f"🎯 Found individual_models - loading {len(individual_models)} models")
            
            for i, model_state in enumerate(individual_models):
                model = PneumoniaClassifier(
                    model_name='efficientnet_b3',
                    num_classes=2,
                    dropout_rate=0.0
                ).to(device)
                
                model.load_state_dict(model_state)
                model.eval()
                models.append(model)
                
        # Method 3: Try direct loading of multiple model keys
        else:
            st.info("🔍 Trying to find individual models with numeric keys...")
            model_count = 0
            
            # Try different possible key patterns
            for i in range(10):  # Check up to 10 models
                possible_keys = [f'model_{i}', f'models.{i}', f'fold_{i}', f'model{i}']
                
                for key_pattern in possible_keys:
                    if key_pattern in checkpoint:
                        model = PneumoniaClassifier(
                            model_name='efficientnet_b3',
                            num_classes=2,
                            dropout_rate=0.0
                        ).to(device)
                        
                        model.load_state_dict(checkpoint[key_pattern])
                        model.eval()
                        models.append(model)
                        model_count += 1
                        break
            
            if model_count == 0:
                st.warning("⚠️ Trying single model loading...")
                # Try loading as single model
                model = PneumoniaClassifier(
                    model_name='efficientnet_b3',
                    num_classes=2,
                    dropout_rate=0.0
                ).to(device)
                
                # Try different single model keys
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                models.append(model)
        
        if len(models) > 0:
            st.success(f"✅ Successfully loaded {len(models)} model(s)!")
            return models, device, checkpoint
        else:
            st.error("❌ No models could be loaded")
            return None, None, None
            
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please check if the model file path is correct and the file is not corrupted.")
        return None, None, None

# FIXED: Ensemble prediction function
def predict_pneumonia_ensemble(models, image, device):
    transform = get_transforms()
    image_np = np.array(image.convert('RGB'))
    
    augmented = transform(image=image_np)
    input_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Get predictions from all models
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for i, model in enumerate(models):
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            all_probabilities.append(probabilities.cpu().numpy()[0])
    
    # Ensemble averaging
    ensemble_probs = np.mean(all_probabilities, axis=0)
    predicted_class = np.argmax(ensemble_probs)
    confidence = ensemble_probs[predicted_class]
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'pneumonia_probability': ensemble_probs[1],
        'normal_probability': ensemble_probs[0],
        'class_name': 'Pneumonia' if predicted_class == 1 else 'Normal',
        'individual_predictions': all_probabilities,
        'num_models': len(models)
    }

# Visualization function
def create_prediction_chart(prediction):
    if prediction['num_models'] == 1:
        # Single model visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        classes = ['Normal', 'Pneumonia']
        probabilities = [prediction['normal_probability'], prediction['pneumonia_probability']]
        colors = ['#28a745', '#dc3545']
        
        bars = ax.bar(classes, probabilities, color=colors, alpha=0.7)
        ax.set_ylabel('Probability')
        ax.set_title('Model Prediction')
        ax.set_ylim(0, 1)
        
        # Add percentage labels
        for bar, prob in zip(bars, probabilities):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{prob:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    else:
        # Multi-model ensemble visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Main prediction chart
        classes = ['Normal', 'Pneumonia']
        probabilities = [prediction['normal_probability'], prediction['pneumonia_probability']]
        colors = ['#28a745', '#dc3545']
        
        bars = ax1.bar(classes, probabilities, color=colors, alpha=0.7)
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Ensemble Prediction ({prediction["num_models"]} Models)')
        ax1.set_ylim(0, 1)
        
        # Add percentage labels
        for bar, prob in zip(bars, probabilities):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{prob:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Individual model predictions
        individual_preds = np.array(prediction['individual_predictions'])
        model_indices = range(1, len(individual_preds) + 1)
        
        ax2.plot(model_indices, individual_preds[:, 1], 'ro-', label='Pneumonia', linewidth=2, markersize=8)
        ax2.plot(model_indices, individual_preds[:, 0], 'go-', label='Normal', linewidth=2, markersize=8)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Probability')
        ax2.set_title('Individual Model Predictions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

# Main App
def main():
    st.markdown('<h1 class="main-header">🫁 Pneumonia Detection AI System</h1>', unsafe_allow_html=True)
    
    # Sidebar info
    st.sidebar.header("📋 About This AI System")
    st.sidebar.info("""
    **World-Class Performance:**
    • 97.5% Validation Accuracy  
    • 0.997 AUC Score
    • 5-Fold Cross Validation
    • EfficientNet-B3 + Attention
    
    **Dataset Used:**
    Chest X-Ray Images (Pneumonia) - Kaggle
    https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
    """)
    
    st.sidebar.header("🔧 System Info")
    device_info = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.text(f"Device: {device_info}")
    
    # Model path input
    st.header("🔍 Model Configuration")
    model_path = st.text_input(
        "📁 **Paste your trained model path here:**",
        value="pneumonia_detection_compl_mod.pth",
        help="Path to your trained .pth model file"
    )
    
    if model_path and model_path != "":
        if st.button("🔄 Load Model", type="primary"):
            with st.spinner("Loading model..."):
                models, device, checkpoint = load_ensemble_model(model_path)
                
                if models is not None:
                    st.success(f"✅ Model loaded successfully! ({len(models)} model(s))")
                    st.session_state['models'] = models
                    st.session_state['device'] = device
                    st.session_state['checkpoint'] = checkpoint
                    
                    # Display model info if available
                    if 'performance_metrics' in checkpoint:
                        st.info("📊 Model Performance Metrics Found!")
                        metrics = checkpoint['performance_metrics']
                        cols = st.columns(4)
                        if 'test_accuracy' in metrics:
                            cols[0].metric("Test Accuracy", f"{metrics['test_accuracy']*100:.1f}%")
                        if 'test_auc' in metrics:
                            cols[1].metric("Test AUC", f"{metrics['test_auc']:.3f}")
                        if 'test_precision' in metrics:
                            cols[2].metric("Test Precision", f"{metrics['test_precision']*100:.1f}%")
                        if 'test_recall' in metrics:
                            cols[3].metric("Test Recall", f"{metrics['test_recall']*100:.1f}%")
    
    # Image upload section
    if 'models' in st.session_state:
        st.header("📷 Upload Chest X-ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.image(image, caption="📸 Uploaded Chest X-ray", use_column_width=True)
            
            with col2:
                st.subheader("📊 Image Information")
                st.write(f"**Format:** {image.format}")
                st.write(f"**Size:** {image.size}")
                st.write(f"**Mode:** {image.mode}")
            
            # Analyze button
            if st.button("🔬 Analyze Image", type="primary"):
                with st.spinner("🤖 AI is analyzing the X-ray..."):
                    prediction = predict_pneumonia_ensemble(
                        st.session_state['models'], 
                        image, 
                        st.session_state['device']
                    )
                    
                    st.header("🎯 Analysis Results")
                    
                    # Display prediction
                    if prediction['predicted_class'] == 0:
                        st.markdown(f"""
                        <div class="prediction-normal">
                            <h3>✅ NORMAL</h3>
                            <p><strong>No signs of pneumonia detected</strong></p>
                            <p>Confidence: {prediction['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-pneumonia">
                            <h3>⚠️ PNEUMONIA DETECTED</h3>
                            <p><strong>Signs of pneumonia found</strong></p>
                            <p>Confidence: {prediction['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence breakdown
                    st.markdown(f"""
                    <div class="confidence-box">
                        <h4>📈 Detailed Confidence Scores</h4>
                        <p><strong>Normal:</strong> {prediction['normal_probability']:.1%}</p>
                        <p><strong>Pneumonia:</strong> {prediction['pneumonia_probability']:.1%}</p>
                        <p><strong>Models Used:</strong> {prediction['num_models']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Visualization
                    st.subheader("📊 Prediction Analysis")
                    fig = create_prediction_chart(prediction)
                    st.pyplot(fig)
                    
                    # Medical recommendation
                    st.header("🏥 Clinical Recommendation")
                    if prediction['predicted_class'] == 1:
                        st.error("""
                        **⚠️ MEDICAL ATTENTION RECOMMENDED**
                        
                        The AI has detected potential pneumonia. Please:
                        • Consult a healthcare professional immediately
                        • Share these results with your doctor
                        • Seek appropriate medical treatment
                        
                        *This AI is for assistance only - not a replacement for professional diagnosis.*
                        """)
                    else:
                        st.success("""
                        **✅ NO IMMEDIATE CONCERNS**
                        
                        No pneumonia signs detected. However:
                        • Continue regular health monitoring
                        • Consult a doctor if symptoms persist
                        • This AI assists but doesn't replace medical check-ups
                        
                        *Always prioritize professional medical advice.*
                        """)
    
    else:
        st.info("👆 Please paste your model path and click 'Load Model' to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown("### 🚨 Important Medical Disclaimer")
    st.warning("""
    This AI system is for **educational and research purposes only**. 
    It should NOT replace professional medical diagnosis or treatment. 
    Always consult qualified healthcare providers for medical decisions.
    """)

if __name__ == "__main__":
    main()
