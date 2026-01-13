# Simulasi Sawit & Banjir

Prototipe simulasi berbasis Python untuk mengeksplorasi dampak perubahan tata guna lahan (sawit vs hutan) terhadap risiko banjir di wilayah Sumatera. Tersedia dua pendekatan pemodelan: agent-based (Mesa) dan discrete-event (SimPy).

---

## 1. Latar Belakang

Indonesia merupakan produsen kelapa sawit terbesar dunia dengan luas perkebunan mencapai lebih dari 16 juta hektar, sebagian besar terkonsentrasi di Sumatera dan Kalimantan (FAO, 2023). Ekspansi perkebunan sawit secara masif telah mengkonversi hutan hujan tropis dan lahan gambut menjadi monokultur, yang berdampak signifikan terhadap siklus hidrologi regional.

Studi terbaru menunjukkan bahwa tutupan hutan tropis, perkebunan sawit, dan curah hujan merupakan faktor pendorong utama kejadian banjir di Provinsi Aceh, dengan dampak sosial-ekonomi yang paling berat dirasakan oleh masyarakat miskin (Lubis et al., 2024). Degradasi lahan gambut akibat pembukaan perkebunan sawit juga menurunkan kemampuan alami gambut dalam meregulasi banjir, namun kebijakan pengelolaan banjir di Indonesia dan Malaysia belum terintegrasi dengan baik ke dalam manajemen gambut (Lupascu et al., 2020). Analisis sistem multi-dimensi gambut tropis Indonesia menunjukkan pentingnya integrasi mitigasi banjir dan regulasi air tawar dalam pengelolaan gambut (Choy & Onuma, 2025).

Konversi hutan ke sawit meningkatkan koefisien aliran permukaan (runoff coefficient) karena penurunan kapasitas infiltrasi tanah dan intersepsi kanopi (Sulistyo et al., 2024). Di Aceh Selatan, penurunan muka tanah (subsidence) akibat pembukaan lahan sawit berkontribusi pada eskalasi risiko banjir (Alfian et al., 2024). Subsidence gambut akibat drainase untuk perkebunan sawit dapat mencapai tingkat yang menjadikan lahan tidak layak untuk budidaya dan meningkatkan risiko banjir musiman (Hein et al., 2022). Analisis spasial-temporal di Musi Rawas, Sumatera Selatan, mengonfirmasi hubungan antara perubahan tutupan lahan sawit dengan peningkatan risiko banjir (Fitra et al., 2025).

Model hidrologi seperti SWAT telah digunakan untuk mensimulasikan dampak perkebunan sawit terhadap kuantitas dan kualitas air di DAS Indonesia (Asmara & Randhir, 2024). Skenario perubahan tutupan lahan juga dapat digunakan untuk memprediksi hazard banjir dan menyusun strategi mitigasi (Putra et al., 2025). Studi jangka panjang di Jambi menunjukkan peningkatan frekuensi dan intensitas banjir seiring konversi hutan ke sawit dan monokultur lainnya (Nguyen, 2025). Dinamika kebakaran, banjir, dan dampak antropogenik pada gambut pesisir dan pedalaman Kalimantan Barat memberikan gambaran kompleksitas interaksi hidrologi-vegetasi-manusia (Ruwaimana et al., 2024).

Praktik pengelolaan air terintegrasi di pertanian gambut tropis dapat mengurangi emisi karbon dan laju subsidence (Fawzi et al., 2024). Evolusi historis reklamasi gambut di Delta Berbak, Jambi, dari padi ke sawit menunjukkan peningkatan risiko banjir akibat subsidence gambut (Widjaja, 2025). Perubahan sifat tanah gambut akibat konversi lahan memberikan implikasi penting untuk restorasi gambut (Kunarso, 2024). Kenaikan muka laut juga mengancam perkebunan sawit pesisir yang sudah rentan akibat subsidence (Panggabean et al., 2025).

Kebutuhan akan alat eksplorasi cepat untuk menilai trade-off antara keuntungan ekonomi sawit dan risiko banjir mendorong pengembangan simulasi berbasis agen dan event-driven yang ringan namun dapat diperluas.

---

## 2. Rumusan Masalah

1. Bagaimana perubahan tutupan lahan (konversi hutan ↔ sawit) memengaruhi kedalaman genangan pada grid lokasi studi?
2. Sejauh mana peningkatan efisiensi drainase dapat menurunkan jumlah sel tergenang pada skenario curah hujan tertentu?
3. Bagaimana kombinasi keputusan tata guna lahan dan infrastruktur drainase memengaruhi reward multi-objektif (profit, penalti banjir, skor biodiversitas) dari waktu ke waktu?

---

## 3. Tujuan

1. Menyediakan lingkungan simulasi ringan berbasis Python yang memadukan pendekatan agent-based modeling (Mesa) dan discrete-event simulation (SimPy).
2. Memungkinkan eksperimen cepat atas skenario tata guna lahan, intensitas curah hujan, dan efisiensi drainase.
3. Menyediakan kerangka yang mudah diperluas ke visualisasi interaktif (Plotly/Mesa server) dan pembungkus reinforcement learning (Gymnasium).

---

## 4. Manfaat

1. **Eksplorasi trade-off**: Membantu memahami pertukaran antara profit sawit, penalti banjir, dan jasa ekosistem.
2. **Sandbox kebijakan**: Platform untuk menguji kebijakan konversi lahan, reforestasi, dan pembangunan drainase secara iteratif sebelum implementasi nyata.
3. **Fondasi integrasi data**: Dapat dihubungkan ke data curah hujan (CHIRPS/GPM IMERG), elevasi (SRTM/Copernicus DEM), dan tutupan lahan (Sentinel-2/OneMap) untuk kalibrasi realistis.
4. **Edukasi dan penelitian**: Dapat digunakan sebagai alat bantu pembelajaran pemodelan hidrologi dan agent-based modeling.

---

## 5. Batasan

1. **Hidrologi disederhanakan**: Tidak ada routing antar-sel, elevasi, atau aliran lateral; hanya akumulasi air per sel.
2. **Curah hujan homogen**: Curah hujan konstan per langkah waktu; belum ada distribusi spasial/temporal atau kurva IDF.
3. **Parameter seragam**: Infiltrasi dan drainase homogen di seluruh grid; belum mengakomodasi variasi tanah/topografi.
4. **Belum divalidasi**: Tidak ada validasi terhadap data observasi lapangan; metrik ekonomi dan ekologi masih skalar sederhana.
5. **Skala terbatas**: Simulasi berskala grid kecil; belum mewakili DAS nyata secara penuh.

---

## 6. Panduan Penggunaan

### 6.1 Prasyarat
- Python 3.10+
- Windows: jalankan perintah di Command Prompt atau PowerShell.

### 6.2 Setup Lingkungan
```bash
# Buat virtual environment
python -m venv .venv

# Aktifkan venv
# CMD:
.venv\Scripts\activate
# PowerShell:
.venv\Scripts\Activate.ps1

# Instal dependensi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set PYTHONPATH
# CMD:
set PYTHONPATH=%CD%\src
# PowerShell:
$env:PYTHONPATH="$PWD/src"
```

### 6.3 Menjalankan Simulasi
```bash
# Mesa (agent-based)
python -m simulasi.main_mesa

# SimPy (discrete-event)
python -m simulasi.main_simpy
```

### 6.4 Struktur Direktori
- `src/simulasi/` — kode sumber simulasi
- `notebooks/` — eksplorasi interaktif
- `docs/` — dokumentasi
- `requirements.txt` — dependensi Python

---

## 7. Referensi

1. Alfian, D., Meilianda, E., & Ahmad, A. (2024). Model the contribution of land subsidence caused by palm oil plantations land clearing to the escalating flood risk in Trumon, South Aceh. *E3S Web of Conferences*. https://doi.org/10.1051/e3sconf/202450701056

2. Asmara, B., & Randhir, T. O. (2024). Modeling the impacts of oil palm plantations on water quantity and quality in the Kais River Watershed of Indonesia. *Science of The Total Environment*. https://doi.org/10.1016/j.scitotenv.2024.026020

3. Choy, Y. K., & Onuma, A. (2025). The tropical peatlands in Indonesia and global environmental change: A multi-dimensional system-based analysis and policy implications. *Regional Science and Environmental Economics*, 2(3), 17. https://doi.org/10.3390/rsee2030017

4. FAO. (2023). *FAOSTAT: Crops and livestock products*. Food and Agriculture Organization of the United Nations.

5. Fawzi, N. I., Sumawinata, B., Rahmasary, A. N., & Qurani, I. Z. (2024). Integrated water management practice in tropical peatland agriculture has low carbon emissions and subsidence rates. *Heliyon*, 10(4), e26616. https://doi.org/10.1016/j.heliyon.2024.e26616

6. Fitra, M. R., Suwignyo, R. A., et al. (2025). Spatial-temporal analysis of land cover change and flood risk detection in North Musi Rawas District, South Sumatra, Indonesia. *Jurnal Locus*.

7. Hein, L., Sumarga, E., Quiñones, M., & Suwarno, A. (2022). Effects of soil subsidence on plantation agriculture in Indonesian peatlands. *Regional Environmental Change*, 22, 131. https://doi.org/10.1007/s10113-022-01979-z

8. Hernanda, T. A. P., Fauzi, A., Barus, B., et al. (2026). Land cover and socio-economic dynamics of coffee to oil palm land conversion in Way Kanan, Indonesia. *Journal of Degraded and Mining Lands Management*.

9. IPCC. (2022). *Climate Change 2022: Impacts, Adaptation and Vulnerability*. Contribution of Working Group II to the Sixth Assessment Report.

10. Kunarso, A. (2024). *Effects of land use change on tropical peat soil properties: Implications for peatland restoration* [Doctoral dissertation, RMIT University].

11. Lubis, M. I., Linkie, M., & Lee, J. S. H. (2024). Tropical forest cover, oil palm plantations, and precipitation drive flooding events in Aceh, Indonesia, and hit the poorest people hardest. *PLOS ONE*. https://doi.org/10.1371/journal.pone.0311759

12. Lupascu, M., Varkkey, H., & Tortajada, C. (2020). Is flooding considered a threat in the degraded tropical peatlands? *Science of The Total Environment*, 723, 137988. https://doi.org/10.1016/j.scitotenv.2020.137988

13. Nguyen, H. C. (2025). *The long-term effects of fire, climate, and human impact in the highland and lowland ecosystems of Sumatra Indonesia* [Doctoral dissertation, University of Göttingen].

14. Panggabean, J., Kurnia, J., & Shaumul, T. (2025). Sea level rise impacts on coastal oil palm plantations. *International Journal of Oil Palm*, 8(1). https://doi.org/10.35876/ijop.v8i1.138

15. Putra, A. N., et al. (2025). Flood prediction: analyzing land use scenarios and strategies in Sumber Brantas and Kali Konto watersheds in East Java, Indonesia. *Natural Hazards*. https://doi.org/10.1007/s11069-025-07363-4

16. Ruwaimana, M., Gavin, D. G., & Anshari, G. (2024). Droughts, fires, floods, and anthropogenic impacts on the peat formation and carbon dynamic of coastal and inland tropical peatlands in West Kalimantan, Indonesia. *Ecosystems*, 27, 45–62. https://doi.org/10.1007/s10021-023-00882-w

17. Sulistyo, B., Adiprasetyo, T., Murcitro, B. G., et al. (2024). Runoff coefficient in the Air Bengkulu Watershed and the evaluation of the existing spatial planning. *The Indonesian Journal of Geography*.

18. Varkkey, H., Ashfold, M., Anshari, G., & Lechner, A. M. (2024). Mitigating carbon emissions and haze in Southeast Asia's peatlands: Opportunities and challenges in integrating policy and governance. In *Tropical Peatland Eco-Management* (pp. 24–48). Springer.

19. Widjaja, H. (2025). From rice to oil palm: The historical evolution of peatland reclamation in the Berbak Delta, Indonesia. *Journal of Tropical Soils*, 30(1). https://doi.org/10.5400/jts.2025.v30i1.671

20. 허수정. (2025). *Disaster Susceptibility Assessment and Management Strategies in Kalimantan and Sumatra Islands, Indonesia* [Master's thesis, Seoul National University].
