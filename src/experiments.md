# Experiments

Deze md bevat alle conceptuele veranderingen aan de source code.

## Overview

De source code omvat 4 stages:

1. Gathering: Het downloaden van data, dit gebeurt voor meeste datasets automatisch, met uitzondering urban atlas en DHM.
2. Processing: Het integreren van data in spatiotemporal query-optimized dataformats
3. Stream: Het gecontroleerd streamen van data, met geregistreerde features
4. ML: Machine learning, toegepast op de stream.

3&4 gebeuren tegelijkertijd, anders zijn deze stages compleet onafhankelijk.

Experiment logs zijn per stage.

---

### Gathering

Gathering is zeer simpel, eerst worden de Gentse grenzen van OpenStreetMap (OSM) verkregen via hun Overpass API, deze stap is gedupliceerd in andere stages. Dan worden datasets gedownload, gefilterd op die grenzen als nodig.

* LST: LST data wordt gedownload vanuit de API, deze is redelijk traag omdat het een website-first soort van API is om blobs te genereren, etc. In essentie moet je eerst zoeken welke image id's er te vinden zijn binnen een tijds- en ruimtelijke span, dan moet je de download links van die image id's krijgen, dan moet je ze downloaden. Dit process kan redelijk lang duren. LST is opgesplits in 3 types: ASTER, MODIS en NDVI. ASTER en MODIS zijn rechtstreeks relateerbare oppervlaktetemperatuurmetingen, NDVI is een vegetatieindex, afhankelijk van temperatuur, dit kan dataleak veroorzaken.
* Trees: boomplantingsdata is een simpele download
* DHM: DHM moet manueel gedownload worden van de website van de vlaamse overheid, en geplaatst in een specifieke folder (zie src/readme)
* Urban Atlas: Hetzelfde geld voor UA (zie src/readme)
* WIS: Het Gentse wegeninformatiesysteem is een simpele download
* G3D: er is ook een gather script voor het 3D model van gent, maar deze is niet in gebruik momenteel

---

#### Gathering experiments

Deze script zijn in 1 keer geschreven en werkten binnen de week met weinig experimentatie, dus er zijn hier geen logs voor geschreven.

---

### Ingestion & Streaming

Deze stage, samen met streaming, waren de grootste uitdaging en zijn extreem afhankelijk van elkaar, dus de experiment logs hiervoor zijn gecombineerd.

Deze is grotendeels conceptueel.

---

#### Ingestion & Streaming experiments

De naïeve oplossing houdt in dat de brondata wordt geïtereerd en features worden gegenereerd daaruit zonder enige optimalizatie.
Met 700M punten in de vergaarde LST data en zeer dure computaties als features, zou dit zeer letterlijk maanden continu duren.

Qua features zijn er meerdere initiële gekozen. Omdat MLR scalar inputs benodigd, en de data conceptueel dat niet is (want LST resolutie is hoger dan DHM, Trees zijn points, polygons, etc.), moeten features aggregaties zijn.

* Trees in radiuses r[]

Om de impact van bomen op LST te bekijken, moeten we enkel features beschouwen die impact hebben op zowel Trees ALS LST, dat zijn dus:

* Elevation statistics (min, max, avg, etc.) (hogere plaatsen zijn waarschijnlijk gebouwen en hebben verschillende weer-condities)
* Land use code coverage in r[] (verschillend landgebruik hebben waarschijnlijk onvermelde bomen = niet geplant door gent)
* Road composition coverage in r[] (zelfde als luc, kan ook inzicht geven in *hoe* een boom best geplant word)

De naïeve manier was eerst geprobeerd, om snel een ML test te kunnen uitvoeren. Maar zelfs voor maar 10000 rows zou dit weken geduurd hebben. Dus werd er besloten om direct een 2-stage pipeline te implementeren. Dit zou tijd afnemen van ML-tests, maar omdat ML uiteindelijk simpele MLR's gingen zijn, is de feature pipeline meer belangrijk.

De eerste gemaakte optimalisatie was overgaan van parquet bestanden naar databases. Om een database te kiezen moet je de soorten data beschouwen:

* LST & DHM .tif's als vervormde grids
* Trees .csv als points
* UA & WIS als polygons

Polygons zijn een zeer niche datastructuur qua grootschalige databases, er is een database nodig die 4*UA+WIS aantal polygons (ongeveer 200k) kan opslaan en efficiënte queries toelaat op de polygon geometrie. We kiezen dus voor spatialite.
Er is geen regel die zegt dat we spatialite ook moeten gebruiken voor de rest, dus gebaseerd op de row-by-row/batch-by-batch streaming requirement, kiezen we de columnar DuckDB.

Deze optimalisatie bespaart niet zo veel tijd, en streams duren nog steeds meerdere seconden per punt. Deze duurde ook enorm lang om te implementeren, omdat polygonal column queries nogal niche zijn en nieuw zijn voor mij en ik was aan het genade van LLM's om mij te helpen met het te verstaan, dit gecombineerd met de grote codebase zorgde voor veel bugs tijdens development die een developer met spatialite ervaring niet zou hebben tegengekomen.

De volgende optimalisatie was het filteren van LST data tot de scope, bij de gather stage werd het echter gefilterd naar "afbeeldingen die deze scope bevatten", dit hield dus ook veel data in buiten die scope. Plus het samenvoegen van ASTER/MODIS/NDVI per image id bij ingestion, ipv spatiotemporal query bij streaming. De filter werd uitgevoerd met een rasterio.mask en de emissivity join bij ingestion adhv de bij gather gedefiniëerde folder names. Dit resulteerde in 200M rows, ipv 700M ASTER/MODIS + 600M NDVI.

Dit resulteert theoretisch gezien in een gigantische speedup, omdat er 6 keer minder rows zijn, 2 keer minder dubbel werk, en dan ook minder werk om NDVI toe te voegen per row.

Uiteindelijk was het niet mogelijk om deze speedup te beschouwen, omdat het nog steeds dagen zou duren voor een dergelijke stream.

De volgende optimalisatie was het precomputen van een raster grid voor polygonal features, om dan ook op "nearest" toe te voegen bij de stream. Deze wordt dan ook gecached.

Dit resulteerde dan in een precompute stage die 20 uur zou duren op een laptop.

