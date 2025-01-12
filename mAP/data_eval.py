from utils.evaluator import get_avg_precision_at_iou
from mAP.data_check import GTDataChecker
import matplotlib.pyplot as plt


def plot_precision_recall_curve(results):
    plt.figure(figsize=(10, 6))
    plt.plot(results['recalls'], results['precisions'], 'b-', label='Precision-Recall curve')
    plt.fill_between(results['recalls'], results['precisions'], alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Average Precision = {results['avg_prec']:.3f}")
    plt.grid(True)
    plt.legend()

    # Додаємо точки на кривій
    plt.scatter(results['recalls'], results['precisions'], c='blue', s=50)

    # Додаємо підписи значень для кожної точки
    for i, (r, p) in enumerate(zip(results['recalls'], results['precisions'])):
        plt.annotate(f'({r:.2f}, {p:.2f})',
                     (r, p),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8)

    # Додаємо анотацію з порогами впевненості
    thresholds_text = "Пороги впевненості:\n"
    for i, thr in enumerate(results['model_thrs']):
        thresholds_text += f"{thr:.2f}"
        if i < len(results['model_thrs']) - 1:
            thresholds_text += ", "
    plt.annotate(thresholds_text,
                 xy=(0.02, 0.02),
                 xycoords='axes fraction',
                 fontsize=8,
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.savefig('precision_recall_curve.png', bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # Завантаження та підготовка даних
    checker = GTDataChecker()
    gt_boxes_all = checker.load_voc_gt()

    # Залишаємо тільки потрібне зображення
    gt_boxes = {'000005': gt_boxes_all['000005']}

    pred_boxes = {
        '000005': {
            'boxes': [[263.0, 211.0, 324.0, 339.0],
                      [165.0, 264.0, 253.0, 372.0],
                      [5.0, 244.0, 67.0, 374.0],
                      [241.0, 194.0, 295.0, 299.0],
                      [277.0, 186.0, 312.0, 220.0]],
            'scores': [0.9, 0.85, 0.82, 0.78, 0.75],  # Оновлені scores
        }
    }

    # Обчислення метрик
    results = get_avg_precision_at_iou(gt_boxes, pred_boxes, iou_thr=0.5)

    # Виведення результатів
    print(f"\nРезультати оцінки:")
    print(f"Середня точність (AP): {results['avg_prec']:.3f}")
    print("\nДетальні результати:")
    print(f"Кількість порогів впевненості: {len(results['model_thrs'])}")
    print("\nЗначення precision та recall для різних порогів впевненості:")
    print("Поріг\t\tPrecision\tRecall")
    print("-" * 40)
    for thr, prec, rec in zip(results['model_thrs'], results['precisions'], results['recalls']):
        print(f"{thr:.3f}\t\t{prec:.3f}\t\t{rec:.3f}")

    # Аналіз результатів
    print("\nАналіз результатів:")
    print(f"1. Найвища precision: {max(results['precisions']):.3f}")
    print(f"2. Найвищий recall: {max(results['recalls']):.3f}")
    print(f"3. Кількість унікальних порогів: {len(set(results['model_thrs']))}")
    print("4. Розподіл scores:")
    print("   - Максимальний score:", max(pred_boxes['000005']['scores']))
    print("   - Мінімальний score:", min(pred_boxes['000005']['scores']))

    # Візуалізація результатів
    plot_precision_recall_curve(results)
    print("\nГрафік збережено у файл 'precision_recall_curve.png'")


if __name__ == "__main__":
    main()