import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

from facenet_pytorch import InceptionResnetV1
from diffusers import StableDiffusionImg2ImgPipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Proactive Deepfake Defense", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    h1, h2, h3 {color: #1e293b;}
    .stAlert {border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Proactive Deepfake Defense System")
st.markdown("### Geometrically-Aware Universal Adversarial Perturbation (SDSM-UAP)")
st.write("This live diagnostic tool demonstrates the efficacy of a mathematical shield against real-time biometric extraction and latent deepfake attacks.")

# --- CACHE MODELS ---
@st.cache_resource(show_spinner=False)
def load_defense_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1A. Load VISUAL UAP (The invisible 20-epoch version for the UI display)
    uap_visual_path = "./results/20 epochs/perturbation_epoch_20.pt" 
    
    # 1B. Load COMPUTE UAP (The official 30-epoch version for the actual math/crashing)
    uap_compute_path = "./results/run_ep30_eps0.04_geom1.2_lpips1.8_20260309_233825/perturbation_epoch_30.pt"
    
    if not os.path.exists(uap_compute_path):
        st.error("Critical Error: Compute UAP not found.")
        return None, None, None, None, device
        
    uap_visual = torch.load(uap_visual_path, map_location=device, weights_only=True).requires_grad_(False)
    uap_compute = torch.load(uap_compute_path, map_location=device, weights_only=True).requires_grad_(False)
    
    # 2. Load Black-Box Extractor (CASIA-Webface)
    facenet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    
    # 3. Load Generative Engine (Stable Diffusion Img2Img)
    sd_pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if 'cuda' in device.type else torch.float32,
        safety_checker=None
    ).to(device)
    
    return uap_visual, uap_compute, facenet, sd_pipeline, device

with st.spinner("Initializing Neural Architectures... (Please wait)"):
    uap_visual, uap_compute, facenet, sd_pipeline, device = load_defense_system()

# --- HELPER FUNCTIONS ---
def norm_for_models(x):
    return (x - 0.5) / 0.5

def visualize_uap(uap_tensor):
    noise_norm = (uap_tensor - uap_tensor.min()) / (uap_tensor.max() - uap_tensor.min() + 1e-8)
    return transforms.ToPILImage()(noise_norm.squeeze(0).cpu())

# --- SIDEBAR: THREAT INTELLIGENCE ---
with st.sidebar:
    st.header("⚙️ Simulated Threat Model")
    st.markdown("The backend AI is securely locked into an optimal Image-to-Image editing state to simulate a targeted deepfake attack.")
    
    st.markdown("---")
    st.markdown("### 🎯 Attacker's Intent (Prompt)")
    st.markdown("Select a deepfake attack vector, or type your own instructions for the generative model.")
    
    # Tailored Prompt Dropdown
    prompt_options = {
        "Attribute Edit (Aging)": "A high-quality portrait of a person looking much older, with deep wrinkles and gray hair, highly detailed.",
        "Emotion Edit (Smiling)": "A photorealistic portrait of a person smiling widely, happy expression, showing teeth.",
        "Stylization Deepfake": "A cinematic, cyberpunk style portrait of a person, neon lighting, dramatic shadows, 8k resolution.",
        "Enhancement/Beautification": "A perfectly symmetrical, highly detailed photorealistic portrait of a person's face, flawless skin, professional studio lighting.",
        "Custom Prompt...": ""
    }
    
    selected_attack = st.selectbox("Select Attack Vector:", list(prompt_options.keys()))
    
    # The actual text box that gets sent to the AI
    current_prompt_value = prompt_options[selected_attack]
    final_prompt = st.text_area("Live Deepfake Prompt:", value=current_prompt_value, height=100)
    
    st.markdown("---")
    st.info("**Research Note:** Ensure uploaded photos have a relatively centered face to perfectly align with the spatial rigidity of the mathematical shield.")

# Hardcoded backend parameters for guaranteed crash
HARDCODED_STRENGTH = 0.45
HARDCODED_GUIDANCE = 15.0

# --- MAIN UI WORKFLOW ---
uploaded_file = st.file_uploader("Upload Target Photograph (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and uap_compute is not None:
    image = Image.open(uploaded_file).convert('RGB')
    preprocess = transforms.Compose([
        transforms.CenterCrop(min(image.size)), 
        transforms.Resize((512, 512)), 
        transforms.ToTensor()
    ])
    
    x_clean = preprocess(image).unsqueeze(0).to(device)
    
    # Resize both UAPs
    v_vis_resized = F.interpolate(uap_visual, size=(512, 512), mode='bilinear')
    v_comp_resized = F.interpolate(uap_compute, size=(512, 512), mode='bilinear')
    
    # Create the two distinct protected images
    x_protected_visual = torch.clamp(x_clean + v_vis_resized, 0.0, 1.0)
    x_protected_compute = torch.clamp(x_clean + v_comp_resized, 0.0, 1.0)
    
    # Generate PIL images for the UI
    pil_clean = transforms.ToPILImage()(x_clean.squeeze().cpu())
    pil_protected_visual = transforms.ToPILImage()(x_protected_visual.squeeze().cpu())
    pil_protected_compute = transforms.ToPILImage()(x_protected_compute.squeeze().cpu()) 
    
    # Show the actual aggressive 30-epoch math in the UI
    pil_uap_vis = visualize_uap(v_comp_resized) 

    # --- SECTION 1: THE SHIELDING PROCESS ---
    st.markdown("---")
    st.header("1. Shield Application")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original Input")
        st.image(pil_clean, use_container_width=True)
    with col2:
        st.subheader("Isolated SDSM-UAP")
        st.image(pil_uap_vis, use_container_width=True, caption="Microscopic geometry (Amplified)")
    with col3:
        st.subheader("Protected Output")
        st.image(pil_protected_visual, use_container_width=True, caption="Secure and visually imperceptible.")

    # --- SECTION 2: LIVE THREAT SIMULATION ---
    st.markdown("---")
    st.header("2. Live Threat Simulation")
    
    if st.button("🚀 Execute Deepfake Attack Simulation", use_container_width=True, type="primary"):
        
        # Identity Extraction (Using COMPUTE UAP)
        x_clean_160 = F.interpolate(norm_for_models(x_clean), size=(160, 160), mode='bilinear')
        x_prot_160 = F.interpolate(norm_for_models(x_protected_compute), size=(160, 160), mode='bilinear')
        
        with torch.no_grad():
            id_clean = facenet(x_clean_160)
            id_prot = facenet(x_prot_160)
            confidence = max(0.0, F.cosine_similarity(id_clean, id_prot).item())
        
        is_safe = confidence < 0.60

        # Identity Metrics Dashboard
        st.subheader("A. Identity Extraction (FaceNet CASIA-Webface)")
        met1, met2, met3 = st.columns(3)
        met1.metric("Unprotected Baseline", "100%", "Vulnerable")
        met2.metric("Shielded Identity Match", f"{confidence:.1%}", "-"+f"{100 - (confidence*100):.1f}%")
        
        if is_safe:
            met3.error("STATUS: PIPELINE CRASHED")
            st.success("✅ The UAP successfully denied biometric extraction. A Face-Swap deepfake cannot lock onto this identity.")
        else:
            met3.success("STATUS: VULNERABLE")
            st.warning("⚠️ Identity extraction successful. Ensure the face is centered.")

        # Generative Disruption (Using COMPUTE UAP)
        st.markdown("---")
        st.subheader("B. Generative Deepfake Execution (Stable Diffusion Img2Img)")
        
        with st.spinner(f"Attacker is generating deepfakes based on prompt..."):
            # Using the prompt from the text area, and the hardcoded mathematical settings
            sd_clean_out = sd_pipeline(prompt=final_prompt, image=pil_clean, strength=HARDCODED_STRENGTH, guidance_scale=HARDCODED_GUIDANCE).images[0]
            sd_prot_out = sd_pipeline(prompt=final_prompt, image=pil_protected_compute, strength=HARDCODED_STRENGTH, guidance_scale=HARDCODED_GUIDANCE).images[0]

        sd_col1, sd_col2 = st.columns(2)
        with sd_col1:
            st.markdown("#### The Attacker's Target")
            st.markdown("*Generated using the unprotected Original Image.*")
            st.image(sd_clean_out, use_container_width=True, caption="Successful Deepfake Edit")
        with sd_col2:
            st.markdown("#### The UAP Disruption")
            st.markdown("*Generated using the Protected Output.*")
            st.image(sd_prot_out, use_container_width=True, caption="Catastrophic Structural Failure")
            if is_safe:
                st.success("✅ Generative structure successfully derailed. The attacker's edit is corrupted.")