# EDA metapodataka i YOLO anotacija

[01 — Dataset EDA](../notebooks/01_dataset_eda.ipynb) prikazuje metapodatke, pacijente,
demografiju, izvorne kodove, sve YOLO klase i integritet veza. Uputstvo za izvršavanje
je u [README-u](../README.md). Za slike i detalje klase fracture pogledajte
[Image i Fracture EDA](image_eda.md).

## Metod

- Identifikatori u CSV-u su tekst kako bi se očuvale vodeće nule. Svaki CSV red ostaje
  u auditu, uključujući prazne, neispravne i dvosmislene identifikatore.
- Anotacioni ZIP daje izvornu mapu klasa i odvojene tabele fajlova, objekata i
  problema. Validacija proverava klasu, broj polja, konačne koordinate, pozitivne
  dimenzije i granice box-a uz toleranciju 1e-6 za izvozno zaokruživanje.
- Prazan validan zapis, validan zapis sa drugim klasama i nepoznat anotacioni
  status prikazani su posebno. Prazne kliničke vrednosti ostaju neupisane.
- Statistike razdvajaju CSV redove, snimke, pacijente i objekte. Najmanji uzrast
  po pacijentu je najmanja zabeležena vrednost. Oznake pola se agregiraju uz
  eksplicitno evidentiranje konflikta.
- Različiti parovi `patient_id` / `study_number` broje se kao identifikatori.
  Njihovo kliničko značenje zahteva potvrdu izvorne dokumentacije.

## Provereni obuhvat

| Metrika | Rezultat |
| --- | ---: |
| CSV redovi / atributi | 20.327 / 17 |
| Pacijenti | 6.091 |
| Različiti parovi pacijent / identifikator studije | 10.699 |
| Jedinstveni slikovni identifikatori | 20.327 |
| YOLO anotacioni fajlovi / klase | 20.327 / 9 |
| Validni YOLO objekti | 47.443 |
| Nevalidni YOLO redovi | 0 |

Oba ZIP inventara i CSV imaju poklapanje identifikatora. Notebook čuva numeričke
sažetke i prazna mesta za lične istraživačke zaključke.
