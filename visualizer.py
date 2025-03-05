import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
    def prepare_plot(segments: list):
        """
        Настройка графика для отображения сегментов с эмоциями.
        segments: список кортежей (start, end, emotion)
        """
        fig, ax = plt.subplots(figsize=(10, 2))
        for start, end, emotion in segments:
            ax.hlines(1, start, end, linewidth=8, label=emotion)
            ax.text((start + end) / 2, 1.1, emotion, ha='center', va='bottom', fontsize=9, rotation=45)
        ax.set_xlim(0, max(end for _, end, _ in segments) + 1)
        ax.set_ylim(0.8, 1.5)
        ax.set_xlabel("Time (s)")
        ax.set_yticks([])
        ax.set_title("Emotion Segments Timeline")
        return fig
