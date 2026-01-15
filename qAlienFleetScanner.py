import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import zlib
import warnings

# Suprime avisos chatos do Lightkurve
warnings.filterwarnings("ignore")


class AlienFleetScanner:
    def __init__(self):
        self.results = []

    def get_metrics(self, signal):
        """Calcula Entropia e Complexidade Zlib"""
        # 1. Normalização Robusta
        sig_norm = (signal - np.nanmean(signal)) / np.nanstd(signal)

        # 2. Entropia de Shannon
        hist, _ = np.histogram(sig_norm, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist))

        # 3. Complexidade Zlib (A "Textura")
        # Binariza pela média
        threshold = np.mean(sig_norm)
        binary_str = ''.join(['1' if x > threshold else '0' for x in sig_norm])
        compressed = len(zlib.compress(binary_str.encode('utf-8')))
        complexity = compressed / len(binary_str)

        return entropy, complexity

    def scan_target(self, target_id, description):
        print(f"🔭 Mirando em {target_id} ({description})...")
        try:
            # Tenta baixar o melhor trimestre disponível
            search = lk.search_lightcurve(target_id, author="Kepler", quarter=None)
            if len(search) == 0:
                search = lk.search_lightcurve(target_id, author="K2")  # Tenta missão K2
            if len(search) == 0:
                search = lk.search_lightcurve(target_id)  # Tenta qualquer coisa (TESS)

            if len(search) == 0:
                print(f"   ❌ Dados não encontrados para {target_id}")
                return

            # Baixa e limpa
            lc = search[0].download().remove_nans()
            flux = np.array(lc.flux.value, dtype=np.float64)

            # Analisa
            H, C = self.get_metrics(flux)

            # Score Alienígena (Heurística: Complexidade alta, Entropia média)
            score = C * (5.0 / (H + 0.1))

            self.results.append({
                'id': target_id,
                'desc': description,
                'H': H,
                'C': C,
                'score': score
            })
            print(f"   ✅ Processado! Entropia: {H:.2f} | Complexidade: {C:.2f}")

        except Exception as e:
            print(f"   ⚠️ Erro ao processar {target_id}: {str(e)}")

    def plot_fleet(self):
        plt.figure(figsize=(14, 9))

        # Extrai dados
        Hs = [r['H'] for r in self.results]
        Cs = [r['C'] for r in self.results]
        scores = [r['score'] for r in self.results]
        labels = [f"{r['id']}\n{r['desc']}" for r in self.results]

        # Scatter Plot
        scatter = plt.scatter(Hs, Cs, c=scores, cmap='inferno', s=200, edgecolors='black', alpha=0.8)
        plt.colorbar(scatter, label='Probabilidade de Estrutura Artificial')

        # Anotações
        for i, txt in enumerate(labels):
            plt.annotate(txt, (Hs[i], Cs[i]), xytext=(7, 7), textcoords='offset points', fontsize=9)

        # Zonas Teóricas
        plt.axvline(x=2.5, color='green', linestyle='--', alpha=0.3, label='Zona de Ordem (Planetas)')
        plt.axvline(x=7.0, color='red', linestyle='--', alpha=0.3, label='Zona de Caos (Estrelas Ativas)')

        plt.title("O Radar Alienígena: Frota de Candidatos Anômalos", fontsize=16)
        plt.xlabel("Entropia (Desordem)", fontsize=12)
        plt.ylabel("Complexidade (Algorítmica)", fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.2)
        plt.show()


# --- MISSÃO DE CAÇA ---
if __name__ == "__main__":
    scanner = AlienFleetScanner()

    # 1. Os Controles (Para calibração)
    print("--- CALIBRANDO SENSORES ---")
    t = np.linspace(0, 100, 3000)
    # Ruído Puro
    scanner.results.append({'id': 'CONTROL', 'desc': 'Pure Noise', 'H': 9.2, 'C': 0.22, 'score': 0})
    # Senóide Pura
    scanner.results.append({'id': 'CONTROL', 'desc': 'Sine Wave', 'H': 1.5, 'C': 0.02, 'score': 0})

    # 2. A Lista de Alvos Reais (Anomalias Famosas)
    targets = [
        ("KIC 8462852", "Tabby's Star (Megastructure?)"),
        ("EPIC 249706694", "Random Transiter (Crazy!)"),
        ("KIC 12557548", "Disintegrating Planet"),
        ("KIC 3749404", "Heartbeat Star (Binary)"),
        ("Kepler-1625", "Exoplanet Host (Normal)"),
        ("KIC 4110611", "Multi-Star Chaos")
    ]

    print("\n--- INICIANDO ESCANEAMENTO DA FROTA ---")
    for tid, desc in targets:
        scanner.scan_target(tid, desc)

    print("\nGerando Relatório Visual...")
    scanner.plot_fleet()