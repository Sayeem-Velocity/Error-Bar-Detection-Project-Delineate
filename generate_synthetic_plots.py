"""
Synthetic Scientific Plot Dataset Generator - With Error Bars

Generates synthetic scientific chart images (PNG) with matching JSON annotations.
ALL plots have error bars with topBarPixelDistance and bottomBarPixelDistance.
Uses matplotlib as the single source of truth for pixel-perfect annotations.
"""

import os
import json
import uuid
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

# Configuration
FIGURE_WIDTH = 7
FIGURE_HEIGHT = 5
DPI = 100
TOTAL_CHARTS = 3000
SEED = 42

# Output directories - fresh generation folder
OUTPUT_BASE = r"D:\Delineate Task'\generated_plots"
OUTPUT_IMAGES = os.path.join(OUTPUT_BASE, "images")
OUTPUT_LABELS = os.path.join(OUTPUT_BASE, "labels")

# Clear existing files if folders exist, then create fresh
import shutil
if os.path.exists(OUTPUT_BASE):
    shutil.rmtree(OUTPUT_BASE)

os.makedirs(OUTPUT_IMAGES, exist_ok=True)
os.makedirs(OUTPUT_LABELS, exist_ok=True)

# Scientific terms
DRUG_NAMES = [
    "Pembrolizumab", "Nivolumab", "Trastuzumab", "Bevacizumab", "Rituximab",
    "Adalimumab", "Infliximab", "Etanercept", "Cetuximab", "Panitumumab",
    "Daratumumab", "Atezolizumab", "Durvalumab", "Ipilimumab", "Avelumab",
    "Vanucizumab", "Ponezumab", "Solanezumab", "Gantenerumab", "Aducanumab"
]

BIOMARKERS = [
    "CRP", "IL-6", "TNF-alpha", "VEGF", "EGF", "HbA1c", "Glucose", "Insulin",
    "C-peptide", "ALT", "AST", "Creatinine", "BUN", "LDL-C", "HDL-C",
    "Troponin-I", "BNP", "WBC", "Hemoglobin", "Platelets", "PSA", "CEA"
]

DOSE_LEVELS = ["0.1mg_kg", "0.3mg_kg", "1mg_kg", "3mg_kg", "10mg_kg", 
               "50mg", "100mg", "200mg", "300mg", "500mg"]

DOSING_REGIMENS = ["QD", "BID", "QW", "Q2W", "Q4W", "single", "multi"]

TREATMENT_GROUPS = ["Placebo", "Control", "Active", "TrtA", "TrtB",
                    "High", "Low", "Mid", "CohortA", "CohortB"]

X_LABELS = ["Time (h)", "Time (days)", "Time (weeks)", "Days", "Weeks",
            "Hours post-dose", "Study Day", "Visit"]

Y_LABELS_CONC = ["Concentration (ng/mL)", "Concentration (ug/mL)", 
                 "Plasma Conc. (ng/mL)", "Serum Conc. (ug/mL)"]

Y_LABELS_BIOMARKER = ["Level (pg/mL)", "Level (ng/mL)", "Level (U/L)",
                      "Percent Change (%)", "Fold Change", "Value"]

# Expanded color palettes for more diversity
COLORS_PRIMARY = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

COLORS_VIBRANT = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']

COLORS_PASTEL = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#f0932b',
                 '#eb4d4b', '#6ab04c', '#686de0', '#30336b', '#95afc0']

COLORS_DARK = ['#2c3e50', '#8e44ad', '#27ae60', '#d35400', '#c0392b',
               '#16a085', '#2980b9', '#7f8c8d', '#f39c12', '#8e44ad']

MARKERS = ['o', 's', '^', 'v', 'D', '<', '>', 'p', 'h', '*', 'X', 'P', 'd', '|', '_']
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]


@dataclass
class PointAnnotation:
    x: float
    y: float
    label: str = ""
    topBarPixelDistance: float = 0
    bottomBarPixelDistance: float = 0
    deviationPixelDistance: float = 0

    def to_dict(self) -> Dict:
        return {
            "x": self.x,
            "y": self.y,
            "label": self.label,
            "topBarPixelDistance": self.topBarPixelDistance,
            "bottomBarPixelDistance": self.bottomBarPixelDistance,
            "deviationPixelDistance": self.deviationPixelDistance
        }


@dataclass
class LineAnnotation:
    lineName: str
    points: List[PointAnnotation] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "label": {"lineName": self.lineName},
            "points": [p.to_dict() for p in self.points]
        }


def generate_line_name(chart_type: str, rng: random.Random) -> str:
    if chart_type == "pk":
        drug = rng.choice(DRUG_NAMES)
        dose = rng.choice(DOSE_LEVELS)
        regimen = rng.choice(DOSING_REGIMENS)
        if rng.random() < 0.4:
            group = rng.choice(TREATMENT_GROUPS)
            return f"{drug}_{dose}_{group}"
        return f"{drug}_{dose}_{regimen}"
    else:
        biomarker = rng.choice(BIOMARKERS)
        group = rng.choice(TREATMENT_GROUPS)
        return f"{biomarker}_{group}"


def generate_pk_data(n_points: int, rng: random.Random, np_rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate PK data with asymmetric error bars"""
    ka = rng.uniform(0.5, 1.5)
    ke = rng.uniform(0.08, 0.2)
    cmax = rng.uniform(100, 400)
    tmax = rng.uniform(2, 6)
    
    # Generate time points
    t = np.sort(np_rng.uniform(0.5, tmax * 5, n_points))
    
    # PK profile
    c = cmax * (np.exp(-ke * t) - np.exp(-ka * t)) * ka / (ka - ke + 0.001)
    c = np.abs(c) + rng.uniform(5, 20)
    c = np.maximum(c, 5)
    
    # Generate asymmetric error bars (can have some zeros for variety)
    top_errors = np.zeros(len(c))
    bottom_errors = np.zeros(len(c))
    
    for i in range(len(c)):
        error_pattern = rng.random()
        
        if error_pattern < 0.15:
            # No error bar at this point (15% chance)
            top_errors[i] = 0
            bottom_errors[i] = 0
        elif error_pattern < 0.35:
            # Symmetric error bars (20% chance)
            err = c[i] * rng.uniform(0.1, 0.3)
            top_errors[i] = err
            bottom_errors[i] = err
        elif error_pattern < 0.55:
            # Top-heavy error (20% chance)
            top_errors[i] = c[i] * rng.uniform(0.15, 0.4)
            bottom_errors[i] = c[i] * rng.uniform(0.05, 0.15)
        elif error_pattern < 0.75:
            # Bottom-heavy error (20% chance)
            top_errors[i] = c[i] * rng.uniform(0.05, 0.15)
            bottom_errors[i] = c[i] * rng.uniform(0.15, 0.35)
        else:
            # Standard CV-based error (25% chance)
            cv = rng.uniform(0.1, 0.25)
            top_errors[i] = c[i] * cv * rng.uniform(0.8, 1.2)
            bottom_errors[i] = c[i] * cv * rng.uniform(0.8, 1.2)
    
    return t, c, top_errors, bottom_errors


def generate_biomarker_data(n_points: int, pattern: str, rng: random.Random, np_rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate biomarker data with asymmetric error bars"""
    max_time = rng.uniform(50, 150)
    t = np.sort(np_rng.uniform(0, max_time, n_points))
    
    if pattern == "increasing":
        baseline = rng.uniform(20, 60)
        slope = rng.uniform(0.3, 1.2)
        y = baseline + slope * t + np_rng.normal(0, baseline * 0.05, n_points)
    elif pattern == "decreasing":
        baseline = rng.uniform(80, 150)
        decay = rng.uniform(0.02, 0.06)
        y = baseline * np.exp(-decay * t) + np_rng.normal(0, baseline * 0.05, n_points)
    elif pattern == "plateau":
        baseline = rng.uniform(30, 60)
        max_val = rng.uniform(100, 180)
        rate = rng.uniform(0.04, 0.1)
        y = max_val - (max_val - baseline) * np.exp(-rate * t)
    else:  # flat/random
        baseline = rng.uniform(50, 120)
        y = baseline + np_rng.normal(0, baseline * 0.1, n_points)
    
    y = np.maximum(y, 5)
    
    # Generate asymmetric error bars
    top_errors = np.zeros(len(y))
    bottom_errors = np.zeros(len(y))
    
    for i in range(len(y)):
        error_pattern = rng.random()
        
        if error_pattern < 0.12:
            top_errors[i] = 0
            bottom_errors[i] = 0
        elif error_pattern < 0.4:
            err = y[i] * rng.uniform(0.08, 0.25)
            top_errors[i] = err
            bottom_errors[i] = err
        elif error_pattern < 0.6:
            top_errors[i] = y[i] * rng.uniform(0.12, 0.35)
            bottom_errors[i] = y[i] * rng.uniform(0.05, 0.12)
        elif error_pattern < 0.8:
            top_errors[i] = y[i] * rng.uniform(0.05, 0.12)
            bottom_errors[i] = y[i] * rng.uniform(0.12, 0.3)
        else:
            cv = rng.uniform(0.08, 0.2)
            top_errors[i] = y[i] * cv * rng.uniform(0.9, 1.1)
            bottom_errors[i] = y[i] * cv * rng.uniform(0.9, 1.1)
    
    return t, y, top_errors, bottom_errors


def data_to_pixel_coords(ax, x_data: np.ndarray, y_data: np.ndarray, fig_height: float) -> List[Tuple[float, float]]:
    pixel_coords = []
    for x, y in zip(x_data, y_data):
        pixel_xy = ax.transData.transform((x, y))
        pixel_coords.append((float(pixel_xy[0]), float(fig_height - pixel_xy[1])))
    return pixel_coords


def error_to_pixel_distance(ax, y_center: float, top_err: float, bottom_err: float) -> Tuple[float, float]:
    pixel_center = ax.transData.transform((0, y_center))
    pixel_top = ax.transData.transform((0, y_center + top_err))
    pixel_bottom = ax.transData.transform((0, max(y_center - bottom_err, 0.1)))
    
    top_dist = abs(pixel_top[1] - pixel_center[1])
    bottom_dist = abs(pixel_center[1] - pixel_bottom[1])
    
    return float(top_dist), float(bottom_dist)


def get_axis_pixel_bounds(ax, fig_height: float) -> Dict[str, Tuple[float, float]]:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    xmin_pixel = ax.transData.transform((xlim[0], ylim[0]))
    xmax_pixel = ax.transData.transform((xlim[1], ylim[0]))
    ymin_pixel = ax.transData.transform((xlim[0], ylim[0]))
    ymax_pixel = ax.transData.transform((xlim[0], ylim[1]))
    
    return {
        "xmin": (float(xmin_pixel[0]), float(fig_height - xmin_pixel[1])),
        "xmax": (float(xmax_pixel[0]), float(fig_height - xmax_pixel[1])),
        "ymin": (float(ymin_pixel[0]), float(fig_height - ymin_pixel[1])),
        "ymax": (float(ymax_pixel[0]), float(fig_height - ymax_pixel[1]))
    }


def generate_chart(chart_idx: int) -> Tuple[str, List[Dict]]:
    rng = random.Random(SEED + chart_idx)
    np_rng = np.random.default_rng(SEED + chart_idx)
    
    # Decide plot style: line (45%), scatter (25%), bar (20%), mixed (10%)
    plot_style = rng.random()
    if plot_style < 0.45:
        style = "line"  # Line/errorbar chart
    elif plot_style < 0.70:
        style = "scatter"  # Scatter plot with error bars
    elif plot_style < 0.90:
        style = "bar"  # Bar chart
    else:
        style = "mixed"  # Mixed styles
    
    # Decide complexity: simple (1 line), medium (2 lines), complex (3-4 lines)
    complexity = rng.random()
    if complexity < 0.4:
        n_lines = 1  # 40% simple
    elif complexity < 0.75:
        n_lines = 2  # 35% medium
    else:
        n_lines = rng.randint(3, 4)  # 25% complex
    
    # Bar charts typically show fewer series
    if style == "bar" and n_lines > 2:
        n_lines = rng.randint(1, 2)
    
    # Scatter plots can be more complex
    if style == "scatter" and rng.random() < 0.3:
        n_lines = rng.randint(3, 5)  # Some scatter plots with many series
    
    chart_type = rng.choice(["pk", "pk", "biomarker"])
    
    # Choose color palette randomly for diversity
    color_palette = rng.choice([COLORS_PRIMARY, COLORS_VIBRANT, COLORS_PASTEL, COLORS_DARK])
    
    # Sometimes use completely random colors
    use_random_colors = rng.random() < 0.25
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT), dpi=DPI)
    fig_height = FIGURE_HEIGHT * DPI
    
    # More diverse styling
    ax.spines['top'].set_visible(rng.random() < 0.4)
    ax.spines['right'].set_visible(rng.random() < 0.4)
    
    # Vary spine thickness
    for spine in ax.spines.values():
        spine.set_linewidth(rng.uniform(0.8, 1.5))
    
    # Grid with more variety
    if rng.random() < 0.7:
        ax.grid(True, alpha=rng.uniform(0.15, 0.5), linestyle=rng.choice(['-', '--', ':', '-.']))
        if rng.random() < 0.3:
            ax.grid(which='minor', alpha=rng.uniform(0.1, 0.2), linestyle=':')
    
    annotations = []
    all_x, all_y = [], []
    
    for line_idx in range(n_lines):
        # Vary number of points based on plot style
        # Scatter plots have more points for visual density
        if style == "scatter":
            n_points = rng.randint(8, 15)
        elif style == "bar":
            n_points = rng.randint(3, 8)
        elif n_lines == 1:
            n_points = rng.randint(5, 10)
        elif n_lines == 2:
            n_points = rng.randint(5, 9)
        else:
            n_points = rng.randint(4, 7)
        
        if chart_type == "pk":
            x_data, y_data, top_errors, bottom_errors = generate_pk_data(n_points, rng, np_rng)
            line_name = generate_line_name("pk", rng)
        else:
            pattern = rng.choice(["increasing", "decreasing", "plateau", "flat"])
            x_data, y_data, top_errors, bottom_errors = generate_biomarker_data(n_points, pattern, rng, np_rng)
            line_name = generate_line_name("biomarker", rng)
        
        all_x.extend(x_data)
        all_y.extend(y_data)
        
        # Diverse color selection
        if use_random_colors:
            # Generate random vibrant color
            color = '#%02x%02x%02x' % (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))
        else:
            color = color_palette[line_idx % len(color_palette)]
        
        marker = rng.choice(MARKERS)
        linestyle = rng.choice(LINE_STYLES)
        markersize = rng.uniform(4, 9)
        linewidth = rng.uniform(1.0, 2.5)
        alpha = rng.uniform(0.7, 1.0)
        
        # Vary error bar style
        capsize = rng.uniform(2, 5)
        capthick = rng.uniform(0.8, 1.5)
        elinewidth = rng.uniform(0.8, 1.5)
        
        # Plot based on style
        if style == "bar":
            # Bar chart with error bars
            bar_width = rng.uniform(0.5, 0.8)
            if n_lines > 1:
                # Group bars side by side
                offset = (line_idx - (n_lines - 1) / 2) * bar_width / n_lines
                x_pos = x_data + offset
            else:
                x_pos = x_data
            
            ax.bar(x_pos, y_data, bar_width / max(n_lines, 1),
                  yerr=[bottom_errors, top_errors],
                  color=color, alpha=alpha,
                  edgecolor=rng.choice(['black', color, 'none']),
                  linewidth=rng.uniform(0.5, 1.5),
                  capsize=capsize,
                  error_kw={'elinewidth': elinewidth, 'capthick': capthick},
                  label=line_name)
        elif style == "scatter":
            # Scatter plot with error bars
            ax.errorbar(x_data, y_data, 
                       yerr=[bottom_errors, top_errors],
                       fmt='o', color=color,
                       linestyle='none',  # No connecting lines
                       markersize=markersize,
                       capsize=capsize,
                       capthick=capthick,
                       elinewidth=elinewidth,
                       alpha=alpha,
                       label=line_name,
                       markeredgewidth=rng.uniform(0.5, 1.5),
                       markerfacecolor=color if rng.random() < 0.6 else 'none',
                       marker=marker)
        else:
            # Line/errorbar chart
            ax.errorbar(x_data, y_data, 
                       yerr=[bottom_errors, top_errors],
                       fmt=marker, color=color,
                       linestyle=linestyle,
                       markersize=markersize,
                       linewidth=linewidth,
                       capsize=capsize,
                       capthick=capthick,
                       elinewidth=elinewidth,
                       alpha=alpha,
                       label=line_name,
                       markeredgewidth=rng.uniform(0.5, 1.5),
                       markerfacecolor=color if rng.random() < 0.7 else 'none')
        
        annotations.append({
            "line_name": line_name,
            "x_data": x_data,
            "y_data": y_data,
            "top_errors": top_errors,
            "bottom_errors": bottom_errors
        })
    
    # Set axis limits
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    x_pad = (x_max - x_min) * rng.uniform(0.05, 0.15)
    y_pad = (y_max - y_min) * rng.uniform(0.08, 0.18)
    
    ax.set_xlim(max(x_min - x_pad, 0), x_max + x_pad)
    ax.set_ylim(max(y_min - y_pad, 0), y_max + y_pad)
    
    # Labels with varied fonts
    xlabel_font = rng.randint(8, 12)
    ylabel_font = rng.randint(8, 12)
    ax.set_xlabel(rng.choice(X_LABELS), fontsize=xlabel_font, weight=rng.choice(['normal', 'bold']))
    if chart_type == "pk":
        ax.set_ylabel(rng.choice(Y_LABELS_CONC), fontsize=ylabel_font, weight=rng.choice(['normal', 'bold']))
    else:
        ax.set_ylabel(rng.choice(Y_LABELS_BIOMARKER), fontsize=ylabel_font, weight=rng.choice(['normal', 'bold']))
    
    # Title (optional) with more variety
    if rng.random() < 0.6:
        if chart_type == "pk":
            title = rng.choice([
                f"{rng.choice(DRUG_NAMES)} PK Profile",
                "Plasma Concentration vs Time",
                "Mean Concentration Profile",
                f"{rng.choice(DOSE_LEVELS)} Dose",
                "Pharmacokinetic Profile"
            ])
        else:
            title = rng.choice([
                f"{rng.choice(BIOMARKERS)} Over Time",
                "Biomarker Response",
                "Mean Biomarker Profile",
                f"{rng.choice(BIOMARKERS)} Levels",
                "Treatment Response"
            ])
        title_font = rng.randint(9, 13)
        ax.set_title(title, fontsize=title_font, weight=rng.choice(['normal', 'bold']), pad=rng.uniform(8, 15))
    
    # Legend with more variety
    show_legend = (n_lines > 1 and rng.random() < 0.85) or (style == "bar" and rng.random() < 0.7)
    if show_legend:
        loc = rng.choice(['best', 'upper right', 'upper left', 'lower right', 'lower left', 'center right'])
        frameon = rng.random() < 0.7
        shadow = rng.random() < 0.3
        ax.legend(loc=loc, fontsize=rng.randint(7, 10), frameon=frameon, shadow=shadow, 
                 fancybox=rng.random() < 0.5, framealpha=rng.uniform(0.7, 1.0))
    
    # Add subtle background color variation
    if rng.random() < 0.15:
        fig.patch.set_facecolor(rng.choice(['#f9f9f9', '#fafafa', '#ffffff', '#f5f5f5']))
        ax.set_facecolor(rng.choice(['#ffffff', '#fafafa', '#f9f9f9']))
    
    # Vary tick parameters for diversity
    ax.tick_params(axis='both', which='major', labelsize=rng.randint(7, 10),
                  length=rng.uniform(3, 6), width=rng.uniform(0.5, 1.2))
    
    plt.tight_layout(pad=rng.uniform(0.5, 1.5))
    fig.canvas.draw()
    
    # Build JSON annotations
    json_annotations = []
    axis_bounds = get_axis_pixel_bounds(ax, fig_height)
    
    for ann in annotations:
        line_ann = LineAnnotation(lineName=ann["line_name"])
        pixel_coords = data_to_pixel_coords(ax, ann["x_data"], ann["y_data"], fig_height)
        
        for i, (px, py) in enumerate(pixel_coords):
            top_err = ann["top_errors"][i]
            bottom_err = ann["bottom_errors"][i]
            
            if top_err > 0 or bottom_err > 0:
                top_dist, bottom_dist = error_to_pixel_distance(ax, ann["y_data"][i], top_err, bottom_err)
            else:
                top_dist, bottom_dist = 0, 0
            
            dev_dist = max(top_dist, bottom_dist)
            
            point = PointAnnotation(
                x=px, y=py, label="",
                topBarPixelDistance=top_dist,
                bottomBarPixelDistance=bottom_dist,
                deviationPixelDistance=dev_dist
            )
            line_ann.points.append(point)
        
        # Add axis bounds
        for label, coord in [("xmin", axis_bounds["xmin"]), ("xmax", axis_bounds["xmax"]),
                            ("ymin", axis_bounds["ymin"]), ("ymax", axis_bounds["ymax"])]:
            line_ann.points.append(PointAnnotation(x=coord[0], y=coord[1], label=label))
        
        json_annotations.append(line_ann.to_dict())
    
    # Save
    file_id = str(uuid.uuid4())
    
    image_path = os.path.join(OUTPUT_IMAGES, f"{file_id}.png")
    fig.savefig(image_path, dpi=DPI, facecolor='white', edgecolor='none',
                bbox_inches='tight', pad_inches=0.1)
    
    json_path = os.path.join(OUTPUT_LABELS, f"{file_id}.json")
    with open(json_path, 'w') as f:
        json.dump(json_annotations, f, indent=2)
    
    plt.close(fig)
    
    return file_id, json_annotations


def main():
    print("=" * 60)
    print("Synthetic Plot Generator - With Error Bars")
    print("=" * 60)
    print(f"Output: {OUTPUT_BASE}")
    print(f"Charts: {TOTAL_CHARTS}")
    print("=" * 60)
    
    successful, failed = 0, 0
    
    for i in range(TOTAL_CHARTS):
        try:
            file_id, _ = generate_chart(i)
            successful += 1
            
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Generated {i + 1}/{TOTAL_CHARTS} (Success: {successful}, Failed: {failed})")
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"Error chart {i}: {e}")
    
    print("=" * 60)
    print(f"Done! Success: {successful}, Failed: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()