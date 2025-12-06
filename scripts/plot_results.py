"""
TS-CVA 실험 결과 시각화 스크립트

사용법:
    python scripts/plot_results.py <ts2vec_dir> <crossmodal_dir> <withnews_dir>
    
예시:
    python scripts/plot_results.py \
        training/tech__ts2vec_only_200ep_20251203_104547_20251203_104554 \
        training/tech__crossmodal_200ep_20251203_104547_20251203_104554 \
        training/tech__crossmodal_news_200ep_20251203_104547_20251203_104554
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 스타일 설정
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'ts2vec': '#2ecc71',      # 초록
    'crossmodal': '#3498db',   # 파랑
    'withnews': '#e74c3c'      # 빨강
}
LABELS = {
    'ts2vec': 'TS2Vec Only',
    'crossmodal': 'Cross-Modal',
    'withnews': 'With News'
}


def load_loss_file(filepath):
    """Loss 파일 로드"""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        lines = f.readlines()
    losses = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                losses.append(float(line))
            except ValueError:
                continue
    return np.array(losses) if losses else None


def load_summary(filepath):
    """Summary 파일 파싱"""
    if not os.path.exists(filepath):
        return None
    
    results = {
        'overall': {},
        'per_symbol': {}
    }
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    section = None
    
    for line in lines:
        line = line.strip()
        
        if 'Overall Results' in line:
            section = 'overall'
            continue
        elif 'Per-Symbol Results' in line:
            section = 'per_symbol'
            continue
        
        if section == 'overall':
            if 'RMSE (raw):' in line:
                results['overall']['rmse'] = float(line.split(':')[1].strip())
            elif 'MAE (raw):' in line:
                results['overall']['mae'] = float(line.split(':')[1].strip())
            elif 'MAPE:' in line:
                results['overall']['mape'] = float(line.split(':')[1].strip().replace('%', ''))
            elif 'Direction Accuracy:' in line:
                results['overall']['dir_acc'] = float(line.split(':')[1].strip().replace('%', ''))
        
        elif section == 'per_symbol':
            if ':' in line and 'MAPE=' in line:
                parts = line.split(':')
                symbol = parts[0].strip()
                metrics_str = parts[1].strip()
                
                metrics = {}
                for metric in metrics_str.split(','):
                    metric = metric.strip()
                    if 'MAPE=' in metric:
                        metrics['mape'] = float(metric.replace('MAPE=', '').replace('%', ''))
                    elif 'MAE=' in metric:
                        metrics['mae'] = float(metric.replace('MAE=', ''))
                    elif 'Dir=' in metric:
                        metrics['dir_acc'] = float(metric.replace('Dir=', '').replace('%', ''))
                
                results['per_symbol'][symbol] = metrics
    
    return results


def plot_loss_curves(dirs, output_dir):
    """Loss 곡선 플롯 - 3가지 모드 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    loss_types = [
        ('forecast_train_loss.txt', 'Forecasting Train Loss (↓ Lower is Better)', 'loss_train_comparison.png'),
        ('forecast_val_loss.txt', 'Forecasting Validation Loss (↓ Lower is Better)', 'loss_val_comparison.png')
    ]
    
    for loss_file, title, filename in loss_types:
        fig, ax = plt.subplots(figsize=(10, 5))
        has_data = False
        for key, dir_path in dirs.items():
            loss = load_loss_file(os.path.join(dir_path, loss_file))
            if loss is not None:
                ax.plot(loss, label=LABELS[key], color=COLORS[key], linewidth=2)
                has_data = True
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (↓ Lower is Better)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        if has_data:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/{filename}")


def plot_individual_loss(dirs, output_dir):
    """각 모드별 개별 Loss 곡선"""
    for key, dir_path in dirs.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        train_loss = load_loss_file(os.path.join(dir_path, 'forecast_train_loss.txt'))
        val_loss = load_loss_file(os.path.join(dir_path, 'forecast_val_loss.txt'))
        
        has_data = False
        if train_loss is not None:
            ax.plot(train_loss, label='Train Loss', color='#3498db', linewidth=2)
            has_data = True
        if val_loss is not None:
            ax.plot(val_loss, label='Validation Loss', color='#e74c3c', linewidth=2)
            has_data = True
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss (↓ Lower is Better)', fontsize=12)
        ax.set_title(f'{LABELS[key]} - Loss Curve (↓ Lower is Better)', fontsize=14, fontweight='bold')
        if has_data:
            ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'loss_{key}.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/loss_{key}.png")


def plot_overall_comparison(summaries, output_dir):
    """전체 성능 비교 바 차트"""
    metrics = ['mape', 'dir_acc', 'rmse', 'mae']
    metric_labels = [
        'MAPE (%) ↓ Lower is Better', 
        'Direction Acc (%) ↑ Higher is Better', 
        'RMSE ↓ Lower is Better', 
        'MAE ↓ Lower is Better'
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = []
        labels = []
        colors = []
        
        for key in ['ts2vec', 'crossmodal', 'withnews']:
            if key in summaries and summaries[key] is not None:
                if metric in summaries[key]['overall']:
                    values.append(summaries[key]['overall'][metric])
                    labels.append(LABELS[key])
                    colors.append(COLORS[key])
        
        if values:
            bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
            
            # 값 표시
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            ax.set_ylabel(label, fontsize=12)
            ax.set_title(label, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/overall_comparison.png")


def plot_per_symbol_comparison(summaries, output_dir):
    """종목별 성능 비교"""
    # 종목 리스트 추출
    symbols = []
    for key, summary in summaries.items():
        if summary is not None and 'per_symbol' in summary:
            symbols = list(summary['per_symbol'].keys())
            break
    
    if not symbols:
        return
    
    metrics = ['mape', 'dir_acc', 'mae']
    metric_labels = [
        'MAPE (%) ↓ Lower is Better', 
        'Direction Accuracy (%) ↑ Higher is Better', 
        'MAE ↓ Lower is Better'
    ]
    
    for metric, label in zip(metrics, metric_labels):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(symbols))
        width = 0.25
        
        for i, (key, offset) in enumerate(zip(['ts2vec', 'crossmodal', 'withnews'], [-width, 0, width])):
            if key in summaries and summaries[key] is not None:
                values = []
                for sym in symbols:
                    if sym in summaries[key]['per_symbol']:
                        values.append(summaries[key]['per_symbol'][sym].get(metric, 0))
                    else:
                        values.append(0)
                
                bars = ax.bar(x + offset, values, width, label=LABELS[key], 
                            color=COLORS[key], edgecolor='white', linewidth=1)
        
        ax.set_xlabel('Symbol', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'Per-Symbol: {label}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(symbols, fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filename = f'per_symbol_{metric}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {output_dir}/{filename}")


def plot_radar_chart(summaries, output_dir):
    """레이더 차트 (종합 성능 비교)"""
    metrics = ['mape', 'dir_acc', 'rmse', 'mae']
    metric_labels = ['MAPE\n(lower better)', 'Direction Acc\n(higher better)', 
                    'RMSE\n(lower better)', 'MAE\n(lower better)']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 닫힌 다각형
    
    for key in ['ts2vec', 'crossmodal', 'withnews']:
        if key in summaries and summaries[key] is not None:
            values = []
            for metric in metrics:
                val = summaries[key]['overall'].get(metric, 0)
                # 정규화 (0-1 범위로)
                if metric == 'dir_acc':
                    val = val / 100  # 높을수록 좋음
                elif metric == 'mape':
                    val = 1 - (val / 20)  # 낮을수록 좋음 (20% 기준)
                elif metric == 'rmse':
                    val = 1 - (val / 30)  # 낮을수록 좋음 (30 기준)
                elif metric == 'mae':
                    val = 1 - (val / 20)  # 낮을수록 좋음 (20 기준)
                values.append(max(0, min(1, val)))  # 0-1 범위 제한
            
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=LABELS[key], color=COLORS[key])
            ax.fill(angles, values, alpha=0.25, color=COLORS[key])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Overall Performance Radar Chart\n(higher is better for all)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/radar_chart.png")


def plot_improvement_heatmap(summaries, output_dir):
    """개선율 히트맵"""
    if 'ts2vec' not in summaries or summaries['ts2vec'] is None:
        return
    
    baseline = summaries['ts2vec']['overall']
    
    improvements = {}
    for key in ['crossmodal', 'withnews']:
        if key in summaries and summaries[key] is not None:
            imp = {}
            for metric in ['mape', 'rmse', 'mae']:
                base_val = baseline.get(metric, 1)
                new_val = summaries[key]['overall'].get(metric, base_val)
                imp[metric] = ((base_val - new_val) / base_val) * 100  # 개선율 (%)
            
            # Direction accuracy는 높을수록 좋음
            base_dir = baseline.get('dir_acc', 50)
            new_dir = summaries[key]['overall'].get('dir_acc', base_dir)
            imp['dir_acc'] = ((new_dir - base_dir) / base_dir) * 100
            
            improvements[key] = imp
    
    if not improvements:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    metrics = ['mape', 'dir_acc', 'rmse', 'mae']
    metric_labels = ['MAPE\n(↓ Lower)', 'Direction Acc\n(↑ Higher)', 'RMSE\n(↓ Lower)', 'MAE\n(↓ Lower)']
    
    data = []
    for key in ['crossmodal', 'withnews']:
        if key in improvements:
            row = [improvements[key].get(m, 0) for m in metrics]
            data.append(row)
    
    data = np.array(data)
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(['crossmodal', 'withnews'])))
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_yticklabels(['Cross-Modal', 'With News'], fontsize=12)
    
    # 값 표시
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            color = 'white' if abs(val) > 10 else 'black'
            sign = '+' if val > 0 else ''
            ax.text(j, i, f'{sign}{val:.1f}%', ha='center', va='center', 
                   color=color, fontsize=12, fontweight='bold')
    
    ax.set_title('Improvement over TS2Vec Only (%)\n(Green = Better, Red = Worse)', 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Improvement (%)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/improvement_heatmap.png")


def save_overall_comparison_csv(summaries, output_dir):
    """전체 성능 비교 CSV 저장"""
    import csv
    
    filepath = os.path.join(output_dir, 'overall_comparison.csv')
    
    metrics = ['mape', 'dir_acc', 'rmse', 'mae']
    headers = ['Model', 'MAPE (%)', 'Direction Accuracy (%)', 'RMSE', 'MAE']
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for key in ['ts2vec', 'crossmodal', 'withnews']:
            if key in summaries and summaries[key] is not None:
                row = [LABELS[key]]
                for metric in metrics:
                    val = summaries[key]['overall'].get(metric, '')
                    row.append(f'{val:.4f}' if val != '' else '')
                writer.writerow(row)
    
    print(f"✅ Saved: {output_dir}/overall_comparison.csv")


def save_per_symbol_comparison_csv(summaries, output_dir):
    """종목별 성능 비교 CSV 저장"""
    import csv
    
    # 종목 리스트 추출
    symbols = []
    for key, summary in summaries.items():
        if summary is not None and 'per_symbol' in summary:
            symbols = list(summary['per_symbol'].keys())
            break
    
    if not symbols:
        return
    
    filepath = os.path.join(output_dir, 'per_symbol_comparison.csv')
    
    # 헤더 생성: Symbol, TS2Vec_MAPE, TS2Vec_DirAcc, TS2Vec_MAE, CrossModal_MAPE, ...
    headers = ['Symbol']
    for key in ['ts2vec', 'crossmodal', 'withnews']:
        headers.extend([
            f'{LABELS[key]}_MAPE(%)',
            f'{LABELS[key]}_DirAcc(%)',
            f'{LABELS[key]}_MAE'
        ])
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        for sym in symbols:
            row = [sym]
            for key in ['ts2vec', 'crossmodal', 'withnews']:
                if key in summaries and summaries[key] is not None:
                    sym_data = summaries[key]['per_symbol'].get(sym, {})
                    row.append(f'{sym_data.get("mape", ""):.4f}' if 'mape' in sym_data else '')
                    row.append(f'{sym_data.get("dir_acc", ""):.4f}' if 'dir_acc' in sym_data else '')
                    row.append(f'{sym_data.get("mae", ""):.4f}' if 'mae' in sym_data else '')
                else:
                    row.extend(['', '', ''])
            writer.writerow(row)
    
    print(f"✅ Saved: {output_dir}/per_symbol_comparison.csv")


def save_improvement_csv(summaries, output_dir):
    """TS2Vec 대비 개선율 CSV 저장"""
    import csv
    
    if 'ts2vec' not in summaries or summaries['ts2vec'] is None:
        return
    
    baseline = summaries['ts2vec']['overall']
    
    filepath = os.path.join(output_dir, 'improvement_over_ts2vec.csv')
    
    headers = ['Model', 'MAPE Improvement (%)', 'DirAcc Improvement (%)', 
               'RMSE Improvement (%)', 'MAE Improvement (%)']
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # Baseline row
        writer.writerow(['TS2Vec Only (Baseline)', '0.00', '0.00', '0.00', '0.00'])
        
        for key in ['crossmodal', 'withnews']:
            if key in summaries and summaries[key] is not None:
                row = [LABELS[key]]
                
                # MAPE, RMSE, MAE: 낮을수록 좋음 (감소하면 양수)
                for metric in ['mape', 'rmse', 'mae']:
                    base_val = baseline.get(metric, 1)
                    new_val = summaries[key]['overall'].get(metric, base_val)
                    if metric == 'mape':
                        # MAPE 먼저 처리
                        imp = ((base_val - new_val) / base_val) * 100
                        row.append(f'{imp:+.2f}')
                    elif metric == 'rmse':
                        imp = ((base_val - new_val) / base_val) * 100
                        # dir_acc 먼저 삽입
                        pass
                    elif metric == 'mae':
                        pass
                
                # 다시 순서대로 계산
                row = [LABELS[key]]
                
                # MAPE
                base_val = baseline.get('mape', 1)
                new_val = summaries[key]['overall'].get('mape', base_val)
                imp = ((base_val - new_val) / base_val) * 100
                row.append(f'{imp:+.2f}')
                
                # Direction Accuracy (높을수록 좋음)
                base_dir = baseline.get('dir_acc', 50)
                new_dir = summaries[key]['overall'].get('dir_acc', base_dir)
                imp = ((new_dir - base_dir) / base_dir) * 100
                row.append(f'{imp:+.2f}')
                
                # RMSE
                base_val = baseline.get('rmse', 1)
                new_val = summaries[key]['overall'].get('rmse', base_val)
                imp = ((base_val - new_val) / base_val) * 100
                row.append(f'{imp:+.2f}')
                
                # MAE
                base_val = baseline.get('mae', 1)
                new_val = summaries[key]['overall'].get('mae', base_val)
                imp = ((base_val - new_val) / base_val) * 100
                row.append(f'{imp:+.2f}')
                
                writer.writerow(row)
    
    print(f"✅ Saved: {output_dir}/improvement_over_ts2vec.csv")


def main():
    if len(sys.argv) != 4:
        print("Usage: python scripts/plot_results.py <ts2vec_dir> <crossmodal_dir> <withnews_dir>")
        print("\nExample:")
        print("  python scripts/plot_results.py \\")
        print("    training/tech__ts2vec_only_200ep_20251203_104547_20251203_104554 \\")
        print("    training/tech__crossmodal_200ep_20251203_104547_20251203_104554 \\")
        print("    training/tech__crossmodal_news_200ep_20251203_104547_20251203_104554")
        sys.exit(1)
    
    dirs = {
        'ts2vec': sys.argv[1],
        'crossmodal': sys.argv[2],
        'withnews': sys.argv[3]
    }
    
    # 디렉토리 확인
    for key, path in dirs.items():
        if not os.path.exists(path):
            print(f"❌ Directory not found: {path}")
            sys.exit(1)
    
    # 출력 디렉토리 생성 (시간 기반)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'plots/{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  TS-CVA Results Visualization")
    print("=" * 60)
    print(f"\nInput directories:")
    for key, path in dirs.items():
        print(f"  [{key}] {path}")
    print(f"\nOutput directory: {output_dir}/")
    print("=" * 60)
    
    # Summary 로드
    summaries = {}
    for key, path in dirs.items():
        summaries[key] = load_summary(os.path.join(path, 'summary.txt'))
    
    # 플롯 생성
    print("\n📊 Generating plots...")
    
    plot_loss_curves(dirs, output_dir)
    plot_individual_loss(dirs, output_dir)
    plot_overall_comparison(summaries, output_dir)
    plot_per_symbol_comparison(summaries, output_dir)
    plot_radar_chart(summaries, output_dir)
    plot_improvement_heatmap(summaries, output_dir)
    
    # CSV 저장
    print("\n📄 Generating CSV files...")
    
    save_overall_comparison_csv(summaries, output_dir)
    save_per_symbol_comparison_csv(summaries, output_dir)
    save_improvement_csv(summaries, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ All plots and CSV files saved successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
