import numpy as np
from scipy import stats
import math


def calculate_deviation_metrics(a, b, c_sequence, threshold_method="sigma", thresholds=None):
    """
    计算C序列相对于参考分布的偏离程度

    Parameters:
    -----------
    a : float
        参考分布的均值
    b : float
        参考分布的标准差
    c_sequence : list or numpy array
        待评估的数据序列
    threshold_method : str
        偏离阈值确定方法："sigma" (标准差倍数) 或 "percentile" (百分位数)
    thresholds : dict or None
        自定义阈值，如果不为None，则覆盖默认阈值

    Returns:
    --------
    dict: 包含各种偏离度量的字典
    """
    c_array = np.array(c_sequence)
    n = len(c_array)

    # 1. 峰值偏离计算
    peak_value = np.max(c_array)  # 峰值
    peak_deviation_sigma = (peak_value - a) / b  # 峰值偏离标准差倍数

    # 2. 平均偏离计算
    # 计算每个元素的偏离标准差倍数
    deviations = (c_array - a) / b

    # 有符号的平均偏离（考虑方向）
    mean_signed_deviation = np.mean(deviations)

    # 无符号的平均偏离（绝对值，只考虑大小）
    mean_absolute_deviation = np.mean(np.abs(deviations))

    # 3. 计算序列的统计特征
    c_mean = np.mean(c_array)
    c_std = np.std(c_array, ddof=1)
    c_skew = stats.skew(c_array)  # 偏度
    c_kurtosis = stats.kurtosis(c_array)  # 峰度

    # 4. 峰值偏离等级评估
    if threshold_method == "percentile":
        # 基于正态分布百分位数评估
        if thresholds is None:
            threshold_peak = {
                'low': 1.0,  # 1σ (68.27% 置信区间)
                'medium': 2.0,  # 2σ (95.45% 置信区间)
                'high': 3.0,  # 3σ (99.73% 置信区间)
                'extreme': 4.0  # 4σ (99.99% 置信区间)
            }
        else:
            threshold_peak = thresholds
    else:  # sigma方法
        if thresholds is None:
            threshold_peak = {
                'low': 1.5,  # 1.5倍标准差
                'medium': 2.5,  # 2.5倍标准差
                'high': 3.5,  # 3.5倍标准差
                'extreme': 5.0  # 5倍标准差
            }
        else:
            threshold_peak = thresholds

    # 评估峰值偏离等级
    peak_abs_dev = abs(peak_deviation_sigma)
    if peak_abs_dev < threshold_peak['low']:
        peak_grade = "正常范围"
        peak_grade_desc = f"峰值在正常波动范围内({peak_abs_dev:.2f}σ)"
    elif peak_abs_dev < threshold_peak['medium']:
        peak_grade = "轻度偏离"
        peak_grade_desc = f"峰值有轻度偏离({peak_abs_dev:.2f}σ)"
    elif peak_abs_dev < threshold_peak['high']:
        peak_grade = "中度偏离"
        peak_grade_desc = f"峰值有中度偏离({peak_abs_dev:.2f}σ)"
    elif peak_abs_dev < threshold_peak['extreme']:
        peak_grade = "显著偏离"
        peak_grade_desc = f"峰值有显著偏离({peak_abs_dev:.2f}σ)"
    else:
        peak_grade = "极端偏离"
        peak_grade_desc = f"峰值有极端偏离({peak_abs_dev:.2f}σ)"

    # 5. 平均偏离等级评估
    mean_abs_dev = mean_absolute_deviation

    if threshold_method == "percentile":
        if thresholds is None:
            threshold_mean = {
                'low': 0.5,  # 0.5σ
                'medium': 1.0,  # 1.0σ
                'high': 1.5,  # 1.5σ
                'extreme': 2.0  # 2.0σ
            }
        else:
            threshold_mean = thresholds
    else:
        if thresholds is None:
            threshold_mean = {
                'low': 0.3,  # 0.3倍标准差
                'medium': 0.6,  # 0.6倍标准差
                'high': 1.0,  # 1.0倍标准差
                'extreme': 1.5  # 1.5倍标准差
            }
        else:
            threshold_mean = thresholds

    if mean_abs_dev < threshold_mean['low']:
        mean_grade = "接近一致"
        mean_grade_desc = f"平均偏离很小({mean_abs_dev:.2f}σ)"
    elif mean_abs_dev < threshold_mean['medium']:
        mean_grade = "轻微偏离"
        mean_grade_desc = f"平均有轻微偏离({mean_abs_dev:.2f}σ)"
    elif mean_abs_dev < threshold_mean['high']:
        mean_grade = "明显偏离"
        mean_grade_desc = f"平均有明显偏离({mean_abs_dev:.2f}σ)"
    elif mean_abs_dev < threshold_mean['extreme']:
        mean_grade = "显著偏离"
        mean_grade_desc = f"平均有显著偏离({mean_abs_dev:.2f}σ)"
    else:
        mean_grade = "严重偏离"
        mean_grade_desc = f"平均有严重偏离({mean_abs_dev:.2f}σ)"

    # 6. 方向性分析
    direction = "偏高" if mean_signed_deviation > 0 else "偏低"
    direction_strength = "强烈" if abs(mean_signed_deviation) > 1.0 else "明显" if abs(
        mean_signed_deviation) > 0.5 else "轻微"

    # 7. 分布形态评估
    distribution_shape = []
    if abs(c_skew) > 0.5:
        distribution_shape.append(f"偏度{'为正' if c_skew > 0 else '为负'}({c_skew:.2f})")
    if c_kurtosis > 1.0:
        distribution_shape.append(f"尖峰态({c_kurtosis:.2f})")
    elif c_kurtosis < -1.0:
        distribution_shape.append(f"平峰态({c_kurtosis:.2f})")

    shape_desc = "、".join(distribution_shape) if distribution_shape else "分布形态基本正常"

    # 8. 综合偏离等级评估
    # 综合考虑峰值偏离、平均偏离和统计显著性
    deviation_score = 0

    # 峰值偏离权重
    if peak_grade == "正常范围":
        deviation_score += 1
    elif peak_grade == "轻度偏离":
        deviation_score += 2
    elif peak_grade == "中度偏离":
        deviation_score += 3
    elif peak_grade == "显著偏离":
        deviation_score += 4
    else:
        deviation_score += 5

    # 平均偏离权重
    if mean_grade == "接近一致":
        deviation_score += 1
    elif mean_grade == "轻微偏离":
        deviation_score += 2
    elif mean_grade == "明显偏离":
        deviation_score += 3
    elif mean_grade == "显著偏离":
        deviation_score += 4
    else:
        deviation_score += 5

    # 方向一致性（如果峰值和均值偏离方向一致，增加分数）
    if np.sign(peak_deviation_sigma) == np.sign(mean_signed_deviation):
        deviation_score += 1

    # 确定综合等级
    if deviation_score <= 3:
        overall_grade = "A级: 高度一致"
        overall_desc = "C序列与参考分布高度一致，属于正常波动范围。"
    elif deviation_score <= 5:
        overall_grade = "B级: 基本一致"
        overall_desc = "C序列与参考分布基本一致，存在轻微但可接受的差异。"
    elif deviation_score <= 7:
        overall_grade = "C级: 中等偏离"
        overall_desc = "C序列与参考分布有中等程度的偏离，需关注但未必异常。"
    elif deviation_score <= 9:
        overall_grade = "D级: 显著偏离"
        overall_desc = "C序列与参考分布有显著偏离，可能存在系统性差异。"
    else:
        overall_grade = "E级: 严重偏离"
        overall_desc = "C序列与参考分布有严重偏离，存在明显异常。"

    # 9. 生成详细文字描述
    detailed_description = f"""
=======================
1. 峰值偏离评估: {peak_grade_desc}
2. 平均偏离评估: {mean_grade_desc}
3. 方向性分析: 序列整体{direction_strength}{direction}于参考均值
4. 分布形态: {shape_desc}
5. 综合评估: {overall_grade} - {overall_desc}
详细数据:
- 参考分布: 均值={a:.4f}, 标准差={b:.4f}
- C序列: 均值={c_mean:.4f}, 标准差={c_std:.4f}, 样本数={n}
- 峰值: {peak_value:.4f} (偏离{peak_deviation_sigma:.4f}σ)
- 平均偏离(有符号): {mean_signed_deviation:.4f}σ
- 平均偏离(绝对值): {mean_abs_dev:.4f}σ
- 偏度: {c_skew:.4f}, 峰度: {c_kurtosis:.4f}
"""

    return {
        # 基本统计
        'reference_mean': a,
        'reference_std': b,
        'sequence_mean': c_mean,
        'sequence_std': c_std,
        'sample_size': n,

        # 峰值偏离
        'peak_value': peak_value,
        'peak_deviation_sigma': peak_deviation_sigma,
        'peak_absolute_deviation': abs(peak_deviation_sigma),
        'peak_grade': peak_grade,
        'peak_grade_description': peak_grade_desc,

        # 平均偏离
        'mean_signed_deviation': mean_signed_deviation,
        'mean_absolute_deviation': mean_absolute_deviation,
        'all_deviations': deviations.tolist(),  # 每个元素的偏离倍数
        'mean_grade': mean_grade,
        'mean_grade_description': mean_grade_desc,

        # 分布特征
        'skewness': c_skew,
        'kurtosis': c_kurtosis,
        'distribution_shape_description': shape_desc,

        # 综合评估
        'deviation_score': deviation_score,
        'overall_grade': overall_grade,
        'overall_description': overall_desc,
        'direction': f"{direction_strength}{direction}",

        # 详细描述
        'detailed_description': detailed_description.strip(),

        # 附加统计量
        'median_deviation': np.median(deviations),
        'std_of_deviations': np.std(deviations, ddof=1),
        'max_negative_deviation': np.min(deviations) if np.any(deviations < 0) else 0,
        'confidence_interval': (
            c_mean - 1.96 * b / np.sqrt(n),
            c_mean + 1.96 * b / np.sqrt(n)
        )
    }


# 可视化函数
def visualize_deviations(deviation_results, save_path=None):
    """
    可视化偏离结果
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. 偏离分布直方图
    ax1 = axes[0, 0]
    deviations = deviation_results['all_deviations']
    ax1.hist(deviations, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='参考均值')
    ax1.axvline(x=deviation_results['mean_signed_deviation'], color='green',
                linestyle='-', linewidth=2, label=f"平均偏离={deviation_results['mean_signed_deviation']:.2f}σ")
    ax1.set_xlabel('偏离标准差倍数(σ)')
    ax1.set_ylabel('频数')
    ax1.set_title('每个元素的偏离分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 偏离程度总结
    ax2 = axes[0, 1]
    metrics = ['峰值偏离', '平均偏离(有符号)', '平均偏离(绝对值)']
    values = [
        deviation_results['peak_deviation_sigma'],
        deviation_results['mean_signed_deviation'],
        deviation_results['mean_absolute_deviation']
    ]
    colors = ['red' if abs(v) > 2 else 'orange' if abs(v) > 1 else 'green' for v in values]
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=-1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=2, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=-2, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_ylabel('偏离标准差倍数(σ)')
    ax2.set_title('偏离程度统计')
    ax2.grid(True, alpha=0.3, axis='y')

    # 在柱子上添加数值标签
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1 * height,
                 f'{val:.2f}σ', ha='center', va='bottom')

    # 3. 等级评估雷达图
    ax3 = axes[1, 0]
    categories = ['峰值偏离', '平均偏离', '方向性', '分布形态']
    scores = [
        5 - list(deviation_results['peak_grade_description'].split()).index(deviation_results['peak_grade']),
        5 - ['接近一致', '轻微偏离', '明显偏离', '显著偏离', '严重偏离'].index(deviation_results['mean_grade']),
        3 if deviation_results['direction'].startswith('强烈') else 2 if deviation_results['direction'].startswith(
            '明显') else 1,
        2 if deviation_results['distribution_shape_description'] == "分布形态基本正常" else 1
    ]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]
    categories_radar = categories + [categories[0]]

    ax3 = plt.subplot(2, 2, 3, polar=True)
    ax3.plot(angles, scores, 'o-', linewidth=2, color='blue')
    ax3.fill(angles, scores, alpha=0.25, color='blue')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(categories)
    ax3.set_ylim(0, 5)
    ax3.set_yticks([1, 2, 3, 4, 5])
    ax3.set_yticklabels(['差', '较差', '中', '良', '优'])
    ax3.set_title('偏离等级评估雷达图')

    # 4. 综合评估文本
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
综合偏离等级: {deviation_results['overall_grade']}
得分: {deviation_results['deviation_score']}/10

详细评估:
1. {deviation_results['peak_grade_description']}
2. {deviation_results['mean_grade_description']}
3. 整体{deviation_results['direction']}
4. {deviation_results['distribution_shape_description']}

统计摘要:
- 序列均值: {deviation_results['sequence_mean']:.4f}
- 序列标准差: {deviation_results['sequence_std']:.4f}
- 偏度: {deviation_results['skewness']:.4f}
- 峰度: {deviation_results['kurtosis']:.4f}
- 95%置信区间: ({deviation_results['confidence_interval'][0]:.4f}, {deviation_results['confidence_interval'][1]:.4f})
"""

    ax4.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('综合评估报告')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    plt.show()


# 使用示例
def analyze_c_sequence_deviation(a, b, c_sequence, visualize=True):
    """
    完整的C序列偏离分析
    """
    # print("=" * 60)
    # print("C序列偏离分析")
    # print("=" * 60)

    # 计算偏离指标
    results = calculate_deviation_metrics(a, b, c_sequence)

    # 打印详细结果
    print(results['detailed_description'])

    # 可视化
    if visualize:
        visualize_deviations(results, save_path='deviation_analysis.png')

    # 返回精简结果
    return {
        'peak_deviation': f"{results['peak_deviation_sigma']:.4f}σ",
        'peak_grade': results['peak_grade'],
        'mean_signed_deviation': f"{results['mean_signed_deviation']:.4f}σ",
        'mean_absolute_deviation': f"{results['mean_absolute_deviation']:.4f}σ",
        'mean_grade': results['mean_grade'],
        'overall_grade': results['overall_grade'],
        'direction': results['direction'],
        'deviation_score': results['deviation_score']
    }


# 快速评估函数
def quick_deviation_assessment(a, b, c_sequence):
    """
    快速评估C序列偏离
    """
    c_array = np.array(c_sequence)

    # 计算峰值偏离
    peak_dev = (np.max(c_array) - a) / b

    # 计算平均偏离
    mean_dev = np.mean((c_array - a) / b)
    mean_abs_dev = np.mean(np.abs(c_array - a) / b)

    # 简单等级评估
    peak_level = "高" if abs(peak_dev) > 3 else "中" if abs(peak_dev) > 2 else "低"
    mean_level = "高" if mean_abs_dev > 1 else "中" if mean_abs_dev > 0.5 else "低"
    direction = "偏高" if mean_dev > 0 else "偏低"

    assessment = f"""
    快速评估结果:
    - 峰值偏离: {abs(peak_dev):.2f}σ ({peak_level}水平)
    - 平均偏离: {mean_abs_dev:.2f}σ ({mean_level}水平)
    - 整体方向: {direction}
    """

    if abs(peak_dev) < 2 and mean_abs_dev < 0.5:
        assessment += "\n结论: 序列与参考分布基本一致，无需特别关注。"
    elif abs(peak_dev) > 3 or mean_abs_dev > 1:
        assessment += "\n结论: 序列与参考分布有显著差异，建议进一步分析。"
    else:
        assessment += "\n结论: 序列存在一定偏离，但尚在可接受范围内。"

    return assessment


# 测试用例
if __name__ == "__main__":
    # 示例1: 正常序列
    np.random.seed(42)
    a, b = 100, 15
    # c_normal = np.random.normal(loc=102, scale=16, size=100)  # 轻微偏离
    #
    # print("示例1: 轻微偏离的序列")
    # print(quick_deviation_assessment(a, b, c_normal))
    # print()
    #
    # # 示例2: 显著偏离的序列
    c_high = np.random.normal(loc=120, scale=20, size=100)  # 显著偏离
    #
    # print("示例2: 显著偏离的序列")
    # print(quick_deviation_assessment(a, b, c_high))
    # print()

    # 完整分析示例
    # results = analyze_c_sequence_deviation(a, b, c_high, visualize=False)
    # print(f"峰值偏离: {results['peak_deviation']} ({results['peak_grade']})")
    # print(f"平均偏离: {results['mean_signed_deviation']} ({results['mean_grade']})")
    # print(f"综合等级: {results['overall_grade']}")
    # print(f"偏离方向: {results['direction']}")

    results = calculate_deviation_metrics(a, b, c_high)
    print(results['detailed_description'])
