# Wrist fracture YOLO research tools

## Eksploratorna analiza podataka

- [01 — Dataset EDA](notebooks/01_dataset_eda.ipynb) · metapodaci, pacijenti, klase i integritet veza.
- [02 — Image EDA](notebooks/02_image_eda.ipynb) · tehnička merenja za sve jednoznačno indeksirane slike i galerije.
- [03 — Fracture EDA](notebooks/03_fracture_eda.ipynb) · učestalost, geometrija, lokalni kontrast i pregled originalnih anotacija.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[eda]"
python -m unittest discover -s tests -v
```

U VS Code-u izaberite projektni `.venv` kernel i pokrenite **Restart Kernel and Run All**
iz korena projekta ili direktorijuma `notebooks/`. Svaki notebook se pokreće samostalno.
Očekivani ulazi u `data/raw/grazpedwri-dx/` su `dataset.csv`, `folder_structure.zip`
i `images_part1.zip` do `images_part4.zip`. ZIP izvori se čitaju direktno.

Prvi prolaz obrađuje slike iterativno i upisuje završene serije u `results/cache/`.
Ponovno izvršavanje nastavlja od nedostajućih serija i koristi validne rezultate.
Podešavanja su u `src/wrist_fracture/data/eda_settings.py`. Detalji metode i pokrivenosti
su u [Dataset EDA vodiču](docs/eda.md) i [Image i Fracture EDA vodiču](docs/image_eda.md).

## Akvizicija podataka

Odeljci koji slede opisuju postojeće akvizicione i inspekcijske komande. Istorijske
beleške niže odnose se na ranije stanje projekta.

## Setup (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Python 3.11+ is required. Runtime dependencies are Requests for HTTP and Pillow
for the existing image inspector. Tests use standard-library unittest. YAML parsing
and Jupyter support are in the optional `[eda]` dependencies. PyTorch and Ultralytics
are deferred to the training stage. Install a hardware-compatible, pinned training environment
when that stage begins. No credentials or `.env` file are needed.

## Inspect and selectively acquire the official dataset

Run from the repository root after installation:

```powershell
wrist-data inspect
wrist-data --version 2 inspect --save data/raw/metadata-v2.json
wrist-data --version 2 download --file dataset.csv --output-dir data/raw/grazpedwri-dx
wrist-data --version 2 download --file dataset.csv --file folder_structure.zip
```

The download examples are instructions, not actions already performed. Use exact
names from the current metadata listing. Inspect lists title, version, every file,
individual byte sizes, total size and URLs, without downloading payloads. `--save`
creates a snapshot and refuses to overwrite an existing one. `--file` is required
and repeatable. There is no implicit download-all or extraction command.

Equivalent module/script entry points:

```powershell
python -m wrist_fracture.data.figshare inspect
python scripts/figshare_dataset.py inspect
wrist-inspect --help
```

After explicitly downloading and safely extracting selected data, use
`wrist-inspect PATH_TO_EXTRACTED_DATA` for read-only inspection. Detailed options
and limitations of that existing inspector remain in the original notes below.

Downloads verify byte size and source MD5 when available, record SHA-256 and full
source metadata under the output directory's `.provenance/`, and reuse verified
completed files. A mismatch fails without replacing originals. Interrupted transfers
restart, with bounded retries; see [the pipeline guide](docs/pipeline.md) for crash
recovery and checksum limitations. Acquisition exits 0 on success, 1 on runtime
failure, and argparse uses 2 for invalid command syntax.

## Current structure

```text
fracture-yolo/
  pyproject.toml                     # package, dependencies, console commands
  README.md                         # setup and preserved research notes
  .gitignore                        # exclude data, weights and generated outputs
  configs/experiments/planned.yaml   # future experiment matrix, not executable yet
  data/                             # local only: raw/, processed/, splits/
  docs/
    pipeline.md                     # architecture, acquisition evidence, future work
    docs/experiment_plan.md         # preserved original research outline
  src/wrist_fracture/
    __init__.py
    data/
      __init__.py
      figshare.py                   # API metadata, selection, transfer, provenance, CLI
      inspect_dataset.py            # preserved read-only local data validation
  scripts/figshare_dataset.py        # thin executable wrapper
  tests/test_figshare.py             # offline acquisition regression tests
  notebooks/                        # future exploration and visualization
  experiments/                      # existing research experiment notes directory
  results/                          # generated run artifacts, ignored by Git
```

Empty directories are not tracked by Git and can be created when first used.
Future training/evaluation/visualization modules are documented rather than filled
with placeholders. Raw data are immutable inputs; processed data are derivatives;
split manifests define reproducible patient assignments. `configs/` tracks settings,
`src/` implements reusable behavior, and `scripts/` provides small entry points.
`results/` will contain unique per-run metrics, configurations, environment records,
plots and checkpoints. Notebooks explore data and call reusable modules.

## Verification and next steps

```powershell
python -m unittest discover -s tests -v
wrist-data --help
wrist-inspect --help
```

Tests use mocked HTTP and temporary files, exercising selection, metadata parsing,
streaming, retries, integrity failures, preservation of existing files, safe names,
locking, provenance and checksum absence. They do not require the dataset or network.
Before a research run, capture `python -m pip freeze` alongside the resolved
configuration, Git state and hardware information; compatibility ranges in
`pyproject.toml` are not an exact environment lock.

The [pipeline guide](docs/pipeline.md) explains the initial repository review,
structural decisions, official API inventory, remote-loading trade-offs, and the
acceptance criteria for data inspection, patient-level splits, preprocessing,
YOLO export, reproducible training and evaluation. Start by explicitly acquiring
metadata and annotations, then inspect real formats and identities before deciding
conversion, negative-image policy or split ratios.

---

## Preserved research notes (Serbian)

The historical sections below describe the project before this acquisition setup;
the operational instructions above and pipeline guide describe the current state.

# Analiza detekcije preloma ručnog zgloba kod dece

Lična dokumentacija za master rad „Analiza uticaja kapaciteta YOLO modela i količine trening podataka na detekciju preloma ručnog zgloba kod dece“.

## Cilj istraživanja

Na datasetu GRAZPEDWRI-DX ispitujem kako kapacitet modela i količina trening podataka utiču na detekciju preloma. Zadatak je object detection sa jednom klasom: `fracture`. Planirani modeli su YOLO11s, YOLO11m i YOLO11l, sa COCO pretrained težinama.

Podela na train, validation i test skup biće na nivou pacijenta: snimci istog pacijenta ne smeju se pojaviti u različitim skupovima. Glavni eksperimenti porede modele i količine trening podataka, dok dodatne analize obuhvataju ulaznu rezoluciju, learning rate, momentum, batch size i augmentacije.

Planirana evaluacija obuhvata mAP@50, mAP@50-95, precision, recall, broj parametara, FLOPs i brzinu inferencije za CPU ONNX i T4 TensorRT. AUC-ROC ostaje opcija samo ako se jasno definiše i metodološki opravda za ovaj zadatak.

## Struktura projekta

Trenutna struktura prikazana je ispod. Folderi su kreirani lokalno, bez `.gitkeep` i drugih placeholder fajlova. Git ne prati prazne foldere, pa oni neće biti preneti kloniranjem repozitorijuma dok ne dobiju sadržaj koji Git prati.

```text
fracture-yolo/
├── .gitignore
├── README.md
├── configs/
│   └── experiments/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── docs/
│   └── docs/
│       └── experiment_plan.md
├── src/
│   └── wrist_fracture/
│       ├── data/
│       │   └── inspect_dataset.py
│       ├── training/
│       ├── evaluation/
│       └── visualization/
├── experiments/
├── notebooks/
└── results/
```

Predloženi naziv projekta je `wrist-fracture-yolo`; lokalni korenski folder za sada ostaje `fracture-yolo`. Naziv budućeg Python paketa je `wrist_fracture`.

`pyproject.toml` (metapodaci projekta i zavisnosti) i `configs/data.yaml` (putanje do skupova i definicija klase) planirani su za trenutak kada uvodimo okruženje i podatke. Još nisu kreirani kako ne bismo unapred dodavali nekorišćene konfiguracije.

| Folder | Namena |
| --- | --- |
| `configs/` | Konfiguracija dataseta kada bude definisana. |
| `configs/experiments/` | Konfiguracije pojedinačnih eksperimenata: model, količina podataka i parametri treninga. |
| `data/raw/` | Izvorni dataset i originalne anotacije, bez menjanja izvornog sadržaja. |
| `data/processed/` | Pripremljeni podaci i anotacije za detekciju jedne klase, kao i izvedeni skupovi za trening i evaluaciju. |
| `data/splits/` | Manifesti patient-level podela i trening podskupova za ponovljive eksperimente. |
| `src/wrist_fracture/data/` | Kod za pripremu podataka, konverziju anotacija i formiranje podela. |
| `src/wrist_fracture/training/` | Kod za trening modela. |
| `src/wrist_fracture/evaluation/` | Kod za evaluaciju, izvoz modela i merenje brzine inferencije. |
| `src/wrist_fracture/visualization/` | Kod za prikaz anotacija, predikcija i poređenja rezultata. |
| `experiments/` | Evidencija sprovedenih eksperimenata i zapažanja, uz reference na korišćene konfiguracije i rezultate. |
| `notebooks/` | Notebook-ovi za istraživanje podataka i analizu rezultata, kada budu potrebni. |
| `results/` | Rezultati evaluacije, poređenja eksperimenata, tabele, grafikoni i merenja brzine inferencije. |

Ako alat za trening bude automatski kreirao `runs/`, tu će ostajati njegovi radni izlazi, logovi i checkpoint-i. Taj folder nije potrebno unapred praviti; odabrani rezultati za analizu pripadaće folderu `results/`.

Postojeći `.gitignore` već isključuje sirove i obrađene podatke, težine modela, ONNX i TensorRT fajlove, `runs/`, `results/`, Python keš i lokalna okruženja. Ovi lokalni artefakti neće biti sačuvani u Git-u. Kod i definicije eksperimenata moći će da se prate kroz Git. Ovaj README ostaje jedina README datoteka projekta.

## Moje beleške

- Dogovoreno: jedna klasa (`fracture`), COCO pretrained YOLO11s/YOLO11m/YOLO11l i patient-level podela podataka.
- Za odluku: odnos train/validation/test skupova, seed i način pouzdanog povezivanja snimaka sa pacijentima.
- Za odluku: nivoi količine trening podataka i način formiranja podskupova; validation i test skupovi treba da ostanu isti kroz poređenja.
- Za proveru: način izdvajanja anotacija preloma i postupanje sa snimcima bez anotiranog preloma.
- Za kasnije: metodološka opravdanost AUC-ROC i precizan protokol merenja FLOPs i brzine inferencije.
- Ideje i nove odluke dopisujem ovde kada dođu na red.

## Inspekcija originalnog dataseta

Skripta `src/wrist_fracture/data/inspect_dataset.py` čita raspakovani dataset i ispisuje izveštaj u terminalu. Ne upisuje fajlove, ne menja anotacije i ne pravi podelu podataka. Dataset trenutno nije dostupan u repozitorijumu, pa stvarne statistike još nisu izračunate.

Pokretanje iz korena projekta:

```powershell
python -B src/wrist_fracture/data/inspect_dataset.py --help
python -B src/wrist_fracture/data/inspect_dataset.py "D:\datasets\GRAZPEDWRI-DX"
```

Pored standardne Python biblioteke koristi se samo Pillow za dekodiranje i proveru slika; već je dostupan u trenutnom okruženju. Ako ga drugo okruženje nema, skripta prijavljuje potrebnu ručnu instalaciju (`python -m pip install Pillow`). Ništa se ne instalira automatski.

Izveštaj obuhvata broj slikovnih fajlova i jedinstvenih naziva, prepoznate pacijente, strukturu foldera, kandidate za formate anotacija, zatečene klase i geometrije, broj objekata po slici, rezolucije, format i bitnu dubinu. Za ispravne boxove prikazuje širinu, visinu, površinu i relativnu površinu kroz minimum, medijanu, p90 i maksimum. Prijavljuje nedostajuće parove slika/anotacija, duplikate, nečitljive slike, neispravne anotacije i neslaganja sa CSV metapodacima. Za svaku vrstu problema ispisuje broj i najviše pet primera.

Podržane slike su PNG, JPEG, TIFF i BMP. Provera dekodira sliku, pa prolazak kroz ceo dataset može potrajati. Kod višestraničnih slika analizira se prva strana uz upozorenje. Neprepoznati fajlovi ostaju u inventaru ekstenzija; arhive se ne raspakuju. Bitna dubina PNG/TIFF slika čita se iz izvornog zaglavlja, a ne samo iz Pillow režima. Za nepodržane režime dubina ostaje nepoznata.

### Izbor anotacija i metapodataka

Automatski izbor daje prednost Supervisely JSON-u, zatim PASCAL VOC XML-u, pa YOLO TXT-u. Analizira se samo jedan izvoz po pokretanju, kako se isti objekti ne bi brojali više puta. Za proveru određenog izvoza koristim, na primer:

```powershell
python -B src/wrist_fracture/data/inspect_dataset.py "D:\datasets\GRAZPEDWRI-DX" --annotation-format voc --annotations "D:\datasets\GRAZPEDWRI-DX\pascalvoc"
```

Za YOLO je obavezno proslediti `--classes` sa svim nazivima klasa redom prema ID-u u metapodacima konkretnog izvoza. Skripta ne pretpostavlja da je `fracture` klasa 0. Automatsko prepoznavanje formata zasniva se na ekstenzijama i nazivima foldera; pri drugačijoj organizaciji treba zadati `--annotations` i format. Neispravan izvoz se ne zamenjuje neprimetno drugim.

Skripta traži jedan `dataset.csv`; alternativu zadajem sa `--metadata`. Kolonu za povezivanje sa slikama bira među `filestem`, `filename` i `file_name`, ili koristim `--filename-column`. Podrazumevane kolone su `patient_id` i `fracture`; druga imena zadajem opcijama `--patient-column` i `--fracture-column`. Neprepoznate vrednosti oznake preloma ostaju nepoznate. Duplirani nazivi slika ili anotacija se prijavljuju i isključuju iz nejednoznačnog povezivanja.

### Identitet pacijenta i izdvajanje fracture klase

Prema [radu autora dataseta](https://www.nature.com/articles/s41597-022-01328-z), početna četiri broja izvornog naziva predstavljaju anonimizovanog pacijenta, npr. `4229` u `4229_0020600143_02_WRI-L1_F010.png`. Dataset sadrži CSV metapodatke i više izvoza anotacija. Skripta koristi CSV identitet kada postoji, a zatim proverava saglasnost sa dokumentovanim obrascem naziva. Ako CSV identitet nedostaje, koristi naziv samo ako odgovara tom obrascu. Konflikti se isključuju iz brojanja identiteta; prijavljuje se pokrivenost, pa broj prepoznatih pacijenata nije nužno konačan broj pacijenata u datasetu.

Broj studija ne izvodi se automatski iz tokena naziva fajla. Tek kada proverim značenje odgovarajuće CSV kolone, mogu da zadam `--study-column IME_KOLONE`; tada se broje parovi pacijent/studija za dostupne identitete. Na stvarnim podacima treba potvrditi da kolona označava pregled, a ne pojedinačni snimak ili seriju.

Za buduće izdvajanje klase koristimo tačan naziv `fracture` u JSON/XML anotacijama, odnosno ID utvrđen iz mape klasa YOLO izvoza. Inspekcija ništa ne filtrira niti konvertuje. Broji i ostale zatečene klase i razlikuje pravougaonike od drugih geometrija. Pre konverzije treba pregledati neispravne boxove i pravilo koordinata izvornog izvoza; trenutne veličine boxova računaju se kao `xmax - xmin` i `ymax - ymin`, bez korekcije za eventualne inkluzivne VOC koordinate.

Slike se razdvajaju na one sa ispravnim fracture boxom, one sa ispravnom anotacijom bez takvog boxa i one sa nepoznatim statusom. Nedostajuće ili neispravne anotacije ne predstavljaju negativne primere. Čak ni ispravna anotacija bez fracture boxa nije automatski negativan nalaz: skripta poredi prisustvo boxa sa CSV oznakom preloma i dostupnim Supervisely tagom. Odsustvo Supervisely taga ostaje nepoznato. Semantiku CSV oznaka i neslaganja treba ručno proveriti pre izbora negativnih primera.

### Provera pokretanja

`--help` proverava CLI bez pristupa datasetu. Pokretanje nad praznim `data/raw/` treba da prijavi da nema slika, bez tvrdnje da je analiza obavljena. Izlazni kod je `0` kada nema prijavljenih problema, `1` kada izveštaj sadrži probleme ili ograničenja, odnosno `2` za nevažeći poziv ili nemogućnost inspekcije. Stvarna validacija obuhvata pokretanje nad preuzetim podacima i ručno poređenje nekoliko slika, anotacija i CSV redova, uključujući prazne anotacije i neslaganja oznaka.

## Evidencija rada

### 2026-10-14 — Priprema strukture

- Pregledan repozitorijum: od projektnih fajlova postojao je samo `.gitignore`.
- Sačuvan `.gitignore` bez izmena jer već pokriva planirane podatke i izlazne artefakte.
- Kreiran `README.md` kao jedina lična dokumentacija projekta, sa ciljevima, planiranom strukturom, beleškama i sledećim koracima.
- Namena budućih foldera dokumentovana umesto kreiranja praznih foldera i placeholder fajlova.
- Nisu kreirane skripte, konfiguracije ni notebook-ovi; nisu instalirane biblioteke niti preuzimani podaci. Funkcionalnost nije implementirana.

### 2026-10-14 — Kreiranje foldera

- Kreirani `data/raw/`, `data/processed/`, `src/`, `experiments/` i `results/` za podatke, kod, definicije eksperimenata i rezultate.
- Ažuriran prikaz strukture u `README.md` tako da odgovara lokalnom stanju projekta.
- Folderi su ostavljeni prazni; `.gitignore` nije menjan i nisu dodavani drugi fajlovi.

### 2026-10-14 — Usklađivanje sa predloženom strukturom

- Kreirani `configs/experiments/`, `data/splits/`, `notebooks/` i podfolderi `data/`, `training/`, `evaluation/` i `visualization/` unutar `src/wrist_fracture/`.
- Ažuriran `README.md` sa namenama foldera i planiranim konfiguracionim fajlovima.
- Sačuvan postojeći `results/` za izlazne rezultate. Lokalni korenski folder nije preimenovan.
- Nisu dodavani placeholder fajlovi, Python kod, notebook-ovi ni konfiguracije; `.gitignore` je ostao neizmenjen.

### 2026-10-21 — Skripta za inspekciju dataseta

- Pregledani postojeći folderi, `README.md` i `docs/docs/experiment_plan.md`; organizacija i plan eksperimenata ostali su sačuvani.
- Kreiran `src/wrist_fracture/data/inspect_dataset.py` za inspekciju slika, anotacija i identiteta pacijenata bez menjanja podataka.
- Dopunjen ovaj README uputstvom za pokretanje, izborom izvora anotacija i ograničenjima interpretacije rezultata. `.gitignore` nije menjan.
- Provereni sintaksa i CLI, parseri Supervisely/VOC/YOLO, dekodiranje 8-bitnih i 16-bitnih PNG slika i odbacivanje oštećene slike. Provereno razdvajanje nedostajućih, praznih i neispravnih anotacija i prijavljivanje neslaganja sa CSV oznakom. Provere koriste podatke samo u memoriji, bez dodatnih testnih fajlova.
- Dataset nije dostupan: analiza stvarnih podataka nije obavljena. Nisu instalirane biblioteke niti pokrenuti preuzimanje, konverzija, split ili trening.

## Sledeći koraci

1. Ručno preuzeti i raspakovati originalni dataset, pa pokrenuti inspekciju.
2. Proveriti CSV kolone, identitete pacijenata, značenje studije i sve prijavljene probleme na stvarnim podacima.
3. Uporediti anotacije i metapodatke za klasu `fracture` i dokumentovati pravilo izbora negativnih primera.
4. Tek nakon završene inspekcije dogovoriti konverziju i buduću patient-level podelu.

Ovi koraci su plan; konverzija, podela podataka i trening nisu započeti.
