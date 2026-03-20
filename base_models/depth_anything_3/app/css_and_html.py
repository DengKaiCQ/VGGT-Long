# flake8: noqa: E501

# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# References:
#   https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/app/css_and_html.py

"""
CSS and HTML content for the Depth Anything 3 Gradio application.
This module contains all the CSS styles and HTML content blocks
used in the Gradio interface.
"""

# CSS Styles for the Gradio interface
GRADIO_CSS = """
/* Add Font Awesome CDN with all styles including brands and colors */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* Add custom styles for colored icons */
.fa-color-blue {
    color: #3b82f6;
}

.fa-color-purple {
    color: #8b5cf6;
}

.fa-color-cyan {
    color: #06b6d4;
}

.fa-color-green {
    color: #10b981;
}

.fa-color-yellow {
    color: #f59e0b;
}

.fa-color-red {
    color: #ef4444;
}

.link-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    text-decoration: none;
    padding: 12px 24px;
    border-radius: 50px;
    font-weight: 500;
    transition: all 0.3s ease;
}

/* Dark mode tech theme */
@media (prefers-color-scheme: dark) {
    html, body {
        background: #1e293b;
        color: #ffffff;
    }

    .gradio-container {
        background: #1e293b;
        color: #ffffff;
    }

    .link-btn {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .link-btn:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
    }

    .tech-bg {
        background: linear-gradient(135deg, #0f172a, #1e293b); /* Darker colors */
        position: relative;
        overflow: hidden;
    }

    .tech-bg::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
            radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.15) 0%, transparent 50%), /* Reduced opacity */
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.15) 0%, transparent 50%), /* Reduced opacity */
            radial-gradient(circle at 40% 40%, rgba(18, 194, 233, 0.1) 0%, transparent 50%); /* Reduced opacity */
        animation: techPulse 8s ease-in-out infinite;
    }

    .gradio-container .panel,
    .gradio-container .block,
    .gradio-container .form {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
    }

    .gradio-container * {
        color: #ffffff;
    }

    .gradio-container label {
        color: #e0e0e0;
    }

    .gradio-container .markdown {
        color: #e0e0e0;
    }
}

/* Light mode tech theme */
@media (prefers-color-scheme: light) {
    html, body {
        background: #ffffff;
        color: #1e293b;
    }

    .gradio-container {
        background: #ffffff;
        color: #1e293b;
    }

    .tech-bg {
        background: linear-gradient(135deg, #ffffff, #f1f5f9);
        position: relative;
        overflow: hidden;
    }

    .link-btn {
        background: rgba(59, 130, 246, 0.15);
        color: var(--body-text-color);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    .link-btn:hover {
        background: rgba(59, 130, 246, 0.25);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
    }

    .tech-bg::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background:
            radial-gradient(circle at 20% 80%, rgba(59, 130, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(18, 194, 233, 0.08) 0%, transparent 50%);
        animation: techPulse 8s ease-in-out infinite;
    }

    .gradio-container .panel,
    .gradio-container .block,
    .gradio-container .form {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .gradio-container * {
        color: #1e293b;
    }

    .gradio-container label {
        color: #334155;
    }

    .gradio-container .markdown {
        color: #334155;
    }
}




@keyframes techPulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 0.8; }
}

/* Custom log with tech gradient */
.custom-log * {
    font-style: italic;
    font-size: 22px !important;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    background-size: 400% 400%;
    -webkit-background-clip: text;
    background-clip: text;
    font-weight: bold !important;
    color: transparent !important;
    text-align: center !important;
    animation: techGradient 3s ease infinite;
}

@keyframes techGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes metricPulse {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes pointcloudPulse {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes camerasPulse {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

@keyframes gaussiansPulse {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Special colors for key terms - Global styles */
.metric-text {
    background: linear-gradient(45deg, #ff6b6b, #ff8e53, #ff6b6b);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    animation: metricPulse 2s ease-in-out infinite;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(255, 107, 107, 0.5);
}

.pointcloud-text {
    background: linear-gradient(45deg, #4ecdc4, #44a08d, #4ecdc4);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    animation: pointcloudPulse 2.5s ease-in-out infinite;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
}

.cameras-text {
    background: linear-gradient(45deg, #667eea, #764ba2, #667eea);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    animation: camerasPulse 3s ease-in-out infinite;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(102, 126, 234, 0.5);
}

.gaussians-text {
    background: linear-gradient(45deg, #f093fb, #f5576c, #f093fb);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
    animation: gaussiansPulse 2.2s ease-in-out infinite;
    font-weight: 700;
    text-shadow: 0 0 10px rgba(240, 147, 251, 0.5);
}

.example-log * {
    font-style: italic;
    font-size: 16px !important;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}

#my_radio .wrap {
    display: flex;
    flex-wrap: nowrap;
    justify-content: center;
    align-items: center;
}

#my_radio .wrap label {
    display: flex;
    width: 50%;
    justify-content: center;
    align-items: center;
    margin: 0;
    padding: 10px 0;
    box-sizing: border-box;
}

/* Align navigation buttons with dropdown bottom */
.navigation-row {
    display: flex !important;
    align-items: flex-end !important;
    gap: 8px !important;
}

.navigation-row > div:nth-child(1),
.navigation-row > div:nth-child(3) {
    align-self: flex-end !important;
}

.navigation-row > div:nth-child(2) {
    flex: 1 !important;
}

/* Make thumbnails clickable with pointer cursor */
.clickable-thumbnail img {
    cursor: pointer !important;
}

.clickable-thumbnail:hover img {
    cursor: pointer !important;
    opacity: 0.8;
    transition: opacity 0.3s ease;
}

/* Make thumbnail containers narrower horizontally */
.clickable-thumbnail {
    padding: 5px 2px !important;
    margin: 0 2px !important;
}

.clickable-thumbnail .image-container {
    margin: 0 !important;
    padding: 0 !important;
}

.scene-info {
    text-align: center !important;
    padding: 5px 2px !important;
    margin: 0 !important;
}
"""


def get_header_html(logo_base64=None):
    """
    Generate the main header HTML with logo and title.

    Args:
        logo_base64 (str, optional): Base64 encoded logo image

    Returns:
        str: HTML string for the header
    """
    return """
    <div class="tech-bg" style="text-align: center; margin-bottom: 5px; padding: 40px 20px; border-radius: 15px; position: relative; overflow: hidden;">
        <div style="position: relative; z-index: 2;">
            <h1 style="margin: 0; font-size: 3.5em; font-weight: 700;
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                background-size: 400% 400%;
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                animation: techGradient 3s ease infinite;
                text-shadow: 0 0 30px rgba(59, 130, 246, 0.5);
                letter-spacing: 2px;">
                Depth Anything 3
            </h1>
            <p style="margin: 15px 0 0 0; font-size: 2.16em; font-weight: 300;" class="header-subtitle">
                Recovering the Visual Space from Any Views
            </p>
            <div style="margin-top: 20px;">
                <!-- Revert buttons to original inline styles -->
                <a href="https://depth-anything-3.github.io" target="_blank" class="link-btn">
                    <i class="fas fa-globe" style="margin-right: 8px;"></i> Project Page
                </a>
                <a href="https://arxiv.org/abs/2406.09414" target="_blank" class="link-btn">
                    <i class="fas fa-file-pdf" style="margin-right: 8px;"></i> Paper
                </a>
                <a href="https://github.com/ByteDance-Seed/Depth-Anything-3" target="_blank" class="link-btn">
                    <i class="fab fa-github" style="margin-right: 8px;"></i> Code
                </a>
            </div>
        </div>
    </div>

    <style>
        /* Ensure tech-bg class is properly applied in dark mode */
        @media (prefers-color-scheme: dark) {
            .header-subtitle {
                color: #cbd5e1;
            }
            /* Increase priority to ensure background color is properly applied */
            .tech-bg {
                background: linear-gradient(135deg, #0f172a, #1e293b) !important;
            }
        }

        @media (prefers-color-scheme: light) {
            .header-subtitle {
                color: #475569;
            }
            /* Also add explicit background color for light mode */
            .tech-bg {
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%) !important;
            }
        }
    </style>
    """


def get_description_html():
    """
    Generate the main description and getting started HTML.

    Returns:
        str: HTML string for the description
    """
    return """
    <div class="description-container" style="padding: 25px; border-radius: 15px; margin: 0 0 20px 0;">
        <h2 class="description-title" style="margin-top: 0; font-size: 1.6em; text-align: center;">
            <i class="fas fa-bullseye fa-color-red" style="margin-right: 8px;"></i> What This Demo Does
        </h2>
        <div class="description-content" style="padding: 20px; border-radius: 10px; margin: 15px 0; text-align: center;">
            <p class="description-main" style="line-height: 1.6; margin: 0; font-size: 1.45em;">
                <strong>Upload images or videos</strong> → <strong>Get <span class="metric-text">Metric</span> <span class="pointcloud-text">Point Clouds</span>, <span class="cameras-text">Cameras</span> and <span class="gaussians-text">Novel Views</span></strong> → <strong>Explore in 3D</strong>
            </p>
        </div>

        <div style="text-align: center; margin-top: 15px;">
            <p class="description-tip" style="font-style: italic; margin: 0;">
                <i class="fas fa-lightbulb fa-color-yellow" style="margin-right: 8px;"></i> <strong>Tip:</strong> Landscape-oriented images or videos are preferred for best 3D recovering.
            </p>
        </div>
    </div>

    <style>
        @media (prefers-color-scheme: dark) {
            .description-container {
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                border: 1px solid rgba(59, 130, 246, 0.2);
            }
            .description-title { color: #3b82f6; }
            .description-content { background: rgba(0, 0, 0, 0.3); }
            .description-main { color: #e0e0e0; }
            .description-text { color: #cbd5e1; }
            .description-tip { color: #cbd5e1; }
        }

        @media (prefers-color-scheme: light) {
            .description-container {
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%);
                border: 1px solid rgba(59, 130, 246, 0.3);
            }
            .description-title { color: #3b82f6; }
            .description-content { background: transparent; }
            .description-main { color: #1e293b; }
            .description-text { color: #475569; }
            .description-tip { color: #475569; }
        }
    </style>
    """


def get_acknowledgements_html():
    """
    Generate the acknowledgements section HTML.

    Returns:
        str: HTML string for the acknowledgements
    """
    return """
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
                padding: 25px; border-radius: 15px; margin: 20px 0; border: 1px solid rgba(59, 130, 246, 0.2);">
        <h3 style="color: #3b82f6; margin-top: 0; text-align: center; font-size: 1.4em;">
            <i class="fas fa-trophy fa-color-yellow" style="margin-right: 8px;"></i> Research Credits & Acknowledgments
        </h3>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 15px 0;">
            <!-- Original Research Section (Left) -->
            <div style="text-align: center;">
                <h4 style="color: #8b5cf6; margin: 10px 0;"><i class="fas fa-flask fa-color-green" style="margin-right: 8px;"></i> Original Research</h4>
                <p style="color: #e0e0e0; margin: 5px 0;">
                    <a href="https://depth-anything-3.github.io" target="_blank"
                       style="color: #3b82f6; text-decoration: none; font-weight: 600;">
                        Depth Anything 3
                    </a>
                </p>
            </div>

            <!-- Previous Versions Section (Right) -->
            <div style="text-align: center;">
                <h4 style="color: #8b5cf6; margin: 10px 0;"><i class="fas fa-history fa-color-blue" style="margin-right: 8px;"></i> Previous Versions</h4>
                <div style="display: flex; flex-direction: row; gap: 15px; justify-content: center; align-items: center;">
                    <p style="color: #e0e0e0; margin: 0;">
                        <a href="https://huggingface.co/spaces/LiheYoung/Depth-Anything" target="_blank"
                           style="color: #3b82f6; text-decoration: none; font-weight: 600;">
                            Depth-Anything
                        </a>
                    </p>
                    <span style="color: #e0e0e0;">•</span>
                    <p style="color: #e0e0e0; margin: 0;">
                        <a href="https://huggingface.co/spaces/depth-anything/Depth-Anything-V2" target="_blank"
                           style="color: #3b82f6; text-decoration: none; font-weight: 600;">
                            Depth-Anything-V2
                        </a>
                    </p>
                </div>
            </div>
        </div>

        <!-- HF Demo Adapted from - Centered at the bottom of the whole block -->
        <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid rgba(59, 130, 246, 0.3); text-align: center;">
            <p style="color: #a0a0a0; font-size: 0.9em; margin: 0;">
                <i class="fas fa-code-branch fa-color-gray" style="margin-right: 5px;"></i> HF demo adapted from <a href="https://huggingface.co/spaces/facebook/map-anything" target="_blank" style="color: inherit; text-decoration: none;">Map Anything</a>
            </p>
        </div>
    </div>
    """


def get_gradio_theme():
    """
    Get the configured Gradio theme with adaptive tech colors.

    Returns:
        gr.themes.Base: Configured Gradio theme
    """
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#eff6ff",
            c100="#dbeafe",
            c200="#bfdbfe",
            c300="#93c5fd",
            c400="#60a5fa",
            c500="#3b82f6",
            c600="#2563eb",
            c700="#1d4ed8",
            c800="#1e40af",
            c900="#1e3a8a",
            c950="#172554",
        ),
        secondary_hue=gr.themes.Color(
            c50="#f5f3ff",
            c100="#ede9fe",
            c200="#ddd6fe",
            c300="#c4b5fd",
            c400="#a78bfa",
            c500="#8b5cf6",
            c600="#7c3aed",
            c700="#6d28d9",
            c800="#5b21b6",
            c900="#4c1d95",
            c950="#2e1065",
        ),
        neutral_hue=gr.themes.Color(
            c50="#f8fafc",
            c100="#f1f5f9",
            c200="#e2e8f0",
            c300="#cbd5e1",
            c400="#94a3b8",
            c500="#64748b",
            c600="#475569",
            c700="#334155",
            c800="#1e293b",
            c900="#0f172a",
            c950="#020617",
        ),
    )


# Measure tab instructions HTML
MEASURE_INSTRUCTIONS_HTML = """
### Click points on the image to compute distance.
> <i class="fas fa-triangle-exclamation fa-color-red" style="margin-right: 5px;"></i> Metric scale estimation is difficult on aerial/drone images.
"""
