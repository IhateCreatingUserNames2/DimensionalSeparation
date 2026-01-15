import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


def generate_signals():
    """Gera os nossos suspeitos habituais"""
    t = np.linspace(0, 100, 1000)

    # 1. Ruído Puro (O Inimigo)
    noise = np.random.normal(0, 1, 1000)

    # 2. Planeta Simples (Ordem)
    planet = np.sin(t)

    # 3. ALIEN DATA (O Sinal "Invisível" da Fase 2)
    # Uma onda portadora modulada com dados densos
    carrier = np.sin(t * 3)
    np.random.seed(42)
    bits = np.random.choice([0, 1], size=1000, p=[0.5, 0.5])
    # Frequency Shift Keying (FSK) - A frequência muda com o bit
    alien = np.sin(t * 3 + (bits * np.pi / 2)) + np.random.normal(0, 0.1, 1000)

    return {"Noise": noise, "Planet": planet, "Alien Signal": alien}


def get_recurrence_plot(signal, threshold_ratio=0.1):
    """
    Cria a Matriz de Recorrência.
    Se o sinal no tempo i é parecido com o tempo j, plota um ponto.
    """
    # Normaliza
    sig = (signal - np.mean(signal)) / np.std(signal)

    # Calcula distância de todos para todos (Euclidiana)
    # Isso cria uma matriz de distância
    dist_matrix = squareform(pdist(sig.reshape(-1, 1), metric='euclidean'))

    # Binariza: Se a distância for pequena (vizinhos), marca 1. Se não, 0.
    threshold = threshold_ratio * np.std(dist_matrix)
    recurrence_matrix = dist_matrix < threshold

    return recurrence_matrix


def calculate_determinism(r_matrix, min_diag_len=2):
    """
    Calcula o DET (Determinismo): % de pontos que formam linhas diagonais.
    Linhas diagonais = Previsibilidade/Regras.
    """
    N = r_matrix.shape[0]
    diagonal_points = 0
    total_points = np.sum(r_matrix)

    # Varre as diagonais para contar linhas
    # (Algoritmo simplificado para demonstração rápida)
    for k in range(1, N):
        diag = np.diagonal(r_matrix, offset=k)
        # Conta sequências de True maiores que min_diag_len
        current_len = 0
        for point in diag:
            if point:
                current_len += 1
            else:
                if current_len >= min_diag_len:
                    diagonal_points += current_len
                current_len = 0
        if current_len >= min_diag_len:  # Conta o final
            diagonal_points += current_len

    if total_points == 0: return 0
    return diagonal_points / total_points


# --- EXECUÇÃO ---
signals = generate_signals()
plt.figure(figsize=(15, 5))

print(f"{'SINAL':<15} | {'DETERMINISMO (DET)':<20} | {'VEREDITO'}")
print("-" * 55)

for i, (name, sig) in enumerate(signals.items()):
    # 1. Calcula Recorrência
    rp = get_recurrence_plot(sig, threshold_ratio=0.2)

    # 2. Calcula Métrica DET
    det_score = calculate_determinism(rp)

    # 3. Classifica
    if det_score < 0.1:
        verdict = "CAOS/RUÍDO"
    elif det_score > 0.9:
        verdict = "SIMPLES/NATURAL"
    else:
        verdict = "COMPLEXIDADE ESTRUTURADA (ALIEN?)"

    print(f"{name:<15} | {det_score:.4f}               | {verdict}")

    # 4. Visualiza
    plt.subplot(1, 3, i + 1)
    plt.imshow(rp, cmap='binary', origin='lower')
    plt.title(f"{name}\nDET: {det_score:.2f}")
    plt.xlabel("Tempo")
    plt.ylabel("Tempo")

plt.tight_layout()
plt.show()