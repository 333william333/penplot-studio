# PenPlot Studio

Ett skrivbordsprogram som gör om **bilder, text och PDF:er** till pennteckningar
på din **Ender 3** – och som också kan **skära stenciler**. Du får en riktig
förhandsvisning av exakt de penndrag som skickas, och kan skicka jobbet direkt
över USB.

Gränssnittet är på engelska (som du valde) och är byggt i Cura-anda: ljust tema,
lager och källa till vänster, förhandsvisning i mitten, inställningar till höger
och en blå knapp längst ner till höger.

Starta från **PenPlot Studio** på skrivbordet, eller från terminalen:

```bash
./run.sh
```

Första gången sätts en Python-miljö upp automatiskt (~5 min). Därefter startar
programmet på ett par sekunder. Ikonen på skrivbordet är ett tunt skal som kör
samma sak – uppdaterar du koden behöver du inte packa om något.

Programmet kan även andra maskiner än Ender 3: 19 färdiga profiler under
*Printer → Machine*, och fyra firmware-familjer (Marlin, Klipper,
RepRapFirmware/Duet, GRBL). Se avsnitt 7b.

---

## 1. Montera pennan

Programmet förutsätter att pennan sitter fast på skrivhuvudet och att **Z-axeln
lyfter pennan**.

* Fäst pennhållaren på hotend-vagnen (buntband, magnethållare eller en tryckt
  hållare fungerar bra).
* Bäst resultat får du med en **fjädrande** upphängning – då spelar det ingen
  roll om bordet lutar en tiondels millimeter.
* Låt pennspetsen sitta **lägre än munstycket**. Munstycket ska aldrig kunna nå
  papperet.
* Tejpa fast papperet på bädden. Ett hörn som lyfter blir ett streck tvärs över
  bilden.

> **Viktigt:** filerna som genereras extruderar aldrig filament och slår aldrig
> på värmen. De hemar heller inte Z – se nästa avsnitt.

## 2. Ställ pennhöjden (görs varje gång du byter penna första gången)

Z-hemning med penna monterad är farligt: pennan trycks ner i bädden. Därför
använder programmet **G92** istället – den höjd pennan står på när jobbet startar
blir Z0.

I fliken **Printer**:

1. Välj USB-porten och tryck **Connect**.
2. Tryck **Set the pen height…** och följ de fyra stegen i guiden:
   hema X och Y → åk till mitten av bädden → sänk Z (5 / 1 / 0,25 / 0,05 mm)
   tills spetsen precis nuddar → **Draw a test line** för att se om trycket är
   rätt → **The pen is touching · use this as Z0**.

Nu står `Drawing Z = 0` och programmet lyfter pennan `Lift` millimeter (2,5 mm
som standard) vid varje förflyttning.

> **Referensen dör med strömmen.** Stäng inte av skrivaren och släpp inte
> motorerna (M84) mellan kalibreringen och **Send**. Gör du det måste höjden
> sättas om. Programmet håller reda på det själv och vägrar starta ett jobb med
> okalibrerad penna – annars hemar skrivaren, lyfter, och ritar hela bilden i
> luften utan att någonsin nudda papperet.

## 3. Arbetsflödet

| Steg | Var | Vad |
|------|-----|-----|
| Källa | Vänster | **Image**, **Text** eller **PDF**. Du kan också släppa en fil rakt på förhandsvisningen. |
| Stil | Vänster | Hur bilden blir till streck – se nedan. |
| Placering | Höger, *Layout* | Storlek, position, rotation, spegling, marginal. |
| Pennor | Höger, *Pens* | Färger, bredder, ordning, pauser. |
| Skrivare | Höger, *Printer* | Z-höjder, hastigheter, bäddstorlek, optimering. |
| Skicka | Nere till höger | **Save G-code…** eller **Send to printer**. |

### Att släppa in en bild

Varje fil du släpper blir **ett eget lager** – den skriver aldrig över det du
redan har – och hamnar centrerad på bädden oavsett var det förra lagret låg.
Släpper du tre filer får du tre lager. Vyn zoomar också till den nya bilden.

Bilden läses av samtidigt: svartpunkt, vitpunkt och gammakurva sätts efter
histogrammet så att det ljusaste blir blankt papper och det mörkaste blir
riktigt svart. Utan det hamnar hela fotot i mitten av tonskalan och skrafferas
jämnt över hela arket – det är det som ser ut som "bara sträck". Är bilden
streckgrafik (en logotyp, en ritning) byter den dessutom till **Sketch**, som
ritar konturerna i stället för att skraffera ytorna.

Har du redan justerat tonen själv rörs ingenting av detta.

### Penntjockleken styr allt

Ändrar du bredden på en penna räknas *hela* bilden om direkt: linjeavstånd,
prickavstånd, hur många konturnivåer som får plats, hur fin detalj som är värd
att rita. Ett 1,2 mm-stift ger inte samma teckning som ett 0,2 mm-stift med
tätare linjer – det ger en annan teckning, gjord för det stiftet.

Reglage som är relativa till pennan säger också vad de betyder i verkligheten:
*"1,00 mm vid en 0,5 mm-penna = 3,00 mm med den här 1,50 mm-pennan."*

### Automatisk inställning

Enklaste vägen till ett snyggt resultat: sätt **Drawing time** till hur länge du
vill att den ska rita och tryck **Auto-tune**. Programmet läser bilden, sätter
svärta och tonkurva efter histogrammet, och justerar sedan täthetsparametern i
en sluten loop tills den faktiskt landar på den tiden. **Pick technique too**
väljer dessutom den teknik som passar bilden bäst.

Träffsäkerheten är ca 10 % median mot måltiden. Kan en teknik inte nå målet
(t.ex. konturlinjer på en enkel bild) säger den det rakt ut i stället för att
låtsas.

### Ingen Render-knapp

Förhandsvisningen ritar om sig själv så fort du ändrar något – reglage,
penntjocklek, teknik, placering. Det finns ingen knapp att trycka på.

Så här hålls den snabb:

* **Väntetiden följer kostnaden.** Fördröjningen innan en omritning startar
  räknas ut från hur lång tid förra bygget tog, så en billig teknik känns
  omedelbar medan en dyr väntar tills du släpper reglaget.
* **Köade byggen kastas.** Drar du ett reglage tolv steg byggs *ett* jobb, inte
  tolv – ett bygge som redan är igång avbryter sig självt när ett nyare kommer.
* **Utkast först.** Tar tekniken mer än ett par tiondelar ritas först ett grovt
  utkast med lägre upplösning, och full kvalitet en stund efter att du slutat
  röra reglaget. Utkastet syns bara på skärmen – det kan aldrig bli en fil.
* **Duken själv ritar 15 gånger snabbare** än förut (samma bild: 78 ms → 5 ms),
  vilket är det som gör att det går att rita om medan du drar.

Sparar eller skickar du mitt i en omritning görs den klar först – du kan aldrig
råka skicka ett utkast eller en gammal version till skrivaren. `⌘R` tvingar fram
en omritning om du vill ha en.

### Navigering i vyn

| | |
|---|---|
| Zooma | rulla, nyp på styrplattan, `+`/`−`, eller knapparna i verktygsraden |
| Panorera | mellanslag + dra, ⌥ + dra, mittenknappen, eller shift + rulla |
| Passa bädden | `F` eller `0` |
| Fyll vyn med ritningen | `Z` eller dubbelklick |

I förhandsvisningen kan du:

* **dra bilden** dit du vill ha den på bädden,
* **dra i hörnhandtagen** för att skala,
* rulla för att zooma, alt-dra eller mitten-dra för att panorera,
* trycka **▶** för att spela upp ritningen i förväg,
* klicka på en penna i teckenförklaringen för att dölja/visa det lagret,
* läsa av sträcka, restid och exakt storlek i rutan nere till vänster.

Rött streckat ramverk = något hamnar utanför bädden.

### Lager – flera saker på samma ark

Överst till vänster ligger **LAYERS**. Ett projekt kan innehålla hur många lager
som helst, och varje lager har sin **egen bild, sin egen teknik och sin egen
placering**: ett foto i crosshatch bredvid en rad text bredvid en handritad ram.

| Knapp | Vad den gör |
|-------|-------------|
| **+ Image / + Text / + PDF** | Nytt lager med den källan |
| **+ Draw** | Tomt ritlager för frihandsstreck |
| Ögat | Visa/dölj lagret – dolda lager ritas inte och räknas inte i tiden |
| ↑ ↓ | Ritordning |
| **Duplicate** | Kopia, förskjuten 10 mm |

Klicka på ett lager i listan – eller direkt på bädden – för att välja det.
Vänsterspalten och *Layout* följer med till det valda lagret, och drar du i
förhandsvisningen är det bara det lagret som flyttas.

### Rita för hand

Verktygsraden ovanför bädden har **frihand, linje, rektangel och ellips**.
Väljer du ett ritverktyg utan att ha ett ritlager markerat skapas ett åt dig.
Strecken sparas i millimeter på bädden precis där du drog dem – ingen skalning,
ingen omplacering. `⌘Z` ångrar det senaste draget.

Ett ritlager har en egen **Tool**-väljare: vilken penna – eller vilket blad –
som ska rita just det lagret.

## 4. Rittekniker

Tjugoen tekniker, grupperade i **Line / Shading / Dots / Geometric**. I vänstra
spalten finns ett **galleri med miniatyrer** som renderas från just din bild –
du ser hur varje teknik skulle bli innan du väljer. Varje teknik har sina egna
parametrar strax under, och knappen *Reset this technique* återställer dem.

### Linjer
| Teknik | Vad den gör |
|--------|-------------|
| **Sketch** | Kantdetektering där varje linje spåras som *ett* drag. Tröskeln sätts automatiskt från bildens gradienter, så den fungerar på både foton och logotyper. Kan få handskakning och överdrag för skisskänsla. |
| **Contour lines** | Höjdkurvor av ljusheten – en sluten linje per nivå. Otroligt fint på porträtt och landskap. |
| **Silhouette** | Spårar konturen av mörka ytor, kan fyllas. Bäst för logotyper, siluetter och text. |
| **Single line** | Hela bilden som **en enda obruten linje**. Punkter placeras efter ton och binds ihop till en optimerad rundtur (2-opt + Or-opt). Noll pennlyft. |

### Skuggning
| Teknik | Vad den gör |
|--------|-------------|
| **Crosshatch** | Lager av snedstreck i olika vinklar. Styrs av **Ink coverage** – hur svart det mörkaste ska bli – och avståndet räknas ut från pennbredden så att det aldrig blir en svart klump. |
| **Pencil dashes** | Brutna streck vars längd följer tonen. Ser ut som snabb blyertsskuggning. |
| **Flow field** | Drag som *följer formerna* i bilden (strukturtensorns riktning). Ger penseldrags- eller gravyrkänsla. Kan även korsa formerna i stället. |
| **Scribble** | En enda vandrande linje som hela tiden söker sig till det bläck som är kvar. Den klassiska klotterporträttseffekten. |
| **Form lines** | Gravyrlinjer som lindar sig *tvärs över* formen. Fler lager läggs bara där det är mörkare, precis som en gravör lägger in en andra och tredje omgång. |
| **Glyph mosaic** | Bilden som ett rutnät av tecken ur det inbyggda enlinjes-typsnittet, valda efter uppmätt bläcktäckning per tecken. Egen teckenuppsättning går att skriva in. |

### Prickar
| Teknik | Vad den gör |
|--------|-------------|
| **Stipple** | Prickar placerade med viktad Lloyd-relaxation (Voronoi via distanstransform) – jämnt fördelade och tonalt korrekta. Prickstorleken kan följa tonen. |
| **Halftone** | Roterat rutnät av cirklar som växer med tonen, som tidningstryck. Flera ringar fyller de mörkaste. |
| **Circle packing** | Cirklar som packas utan att överlappa – stora i ljusa partier, små och täta i mörka. Går att vända. |
| **Dwell dots** | Jämnt rutnät där **mörkheten kommer från hur länge pennan står still**, inte från prickens storlek. Se nedan. |

### Geometriskt
| Teknik | Vad den gör |
|--------|-------------|
| **Voronoi cells** | Punkter fördelade efter ton, sedan cellgränserna. Organisk struktur som tätnar i skuggorna. |
| **Maze** | En riktig labyrint över hela arket, men väggarna ritas bara där bilden har bläck. |
| **Spiral** | En enda spiral från mitten som vippar där bilden är mörk. |
| **Concentric rings** | Slutna ringar i stället för spiral – renare och mer grafiskt. |
| **Wave lines** | Parallella vågor vars amplitud följer tonen. |
| **Hilbert curve** | En enda kontinuerlig rymdfyllande linje som viker sig tätare i mörka partier. *Base grid* styr hur fin den är även på vitt papper. |
| **Triangle mesh** | Punkter fördelade efter ton, sammanbundna till ett Delaunay-nät. Low-poly-look. |

Alla avstånd anges i **millimeter på papperet**, aldrig i pixlar, och de
pennbreddsberoende (linjeavstånd, prickavstånd) skalas automatiskt från en
0,5 mm-referenspenna.

### Dwell dots – tid i stället för storlek

Med en fiberpenna, reservoarpenna, gelpenna eller tuschpenna fortsätter bläcket
att sugas ut i papperet så länge spetsen vilar. En prick på 400 ms blir därför
synligt mörkare och fetare än en på 20 ms. Den tekniken ger en riktig gråskala
ur **en enda penna** utan att ändra geometrin – programmet lägger ett `G4`-
väntekommando efter varje pennsänkning och grupperar prickarna i några få
dwell-steg så filen inte sväller.

**Det fungerar inte med kulspetspenna** – den behöver rörelse för att skriva
alls. Därför har varje penna ett **Tip**-fält under Pens (fiberspets, reservoar,
gel, tusch, kulspets, blyerts). Pennor som bläder får en ◍-markering i listan,
och väljer du Dwell dots med en penna som inte kan det får du en tydlig
varning i vänsterspalten i stället för ett tråkigt resultat.

### Ton från maskinen, inte från geometrin

Under **Technique settings → Tone from the machine** kan vilken teknik som helst
få varierande linjevikt utan att ett enda drag läggs till:

* **Pen pressure (Z)** – pennan pressas upp till 0,6 mm djupare där bilden är
  mörk. Ger tjockare, mörkare linje precis som när man trycker hårdare för hand.
  Kräver **fjädrande pennhållare** – programmet varnar för det.
* **Drawing speed** – maskinen saktar ner där bilden är mörk. Filtpennor och
  reservoarpennor lämnar då mer bläck. Kulspets påverkas knappt.

Samma antal drag, samma ritt tid – men ett mycket rikare resultat. Z-värdet
skrivs in i varje ritkommando och följer med live-reglaget för pennanslag.

## 5. Flera färger och flera pennor

Under **Pens** laddar du ett pennset (eller bygger ett eget). Under **Colours**
i vänsterspalten väljer du hur bilden delas upp:

* **Single pen (grayscale)** – allt med en penna.
* **Match my pen colours** – varje pixel hamnar på den penna som ligger närmast
  i färg (jämförs i CIE-Lab), och mängden bläck följer hur mörk pixeln är.
  Knappen **Suggest pen colours from the picture** k-means:ar fram en palett
  ur bilden åt dig.
* **CMYK separation** – klassisk fyrfärgsseparation.

Varje penna ritas som ett eget lager, i listans ordning. Knappen
**Order light → dark** sorterar ljusast först så att mörkt bläck aldrig dras
genom vått ljust bläck.

**Pennbredden styr genereringen.** Inställningarna är skrivna för en 0,5 mm-penna
och skalas därifrån: en 1,0 mm-penna får dubbelt linjeavstånd, en 0,25 mm-penna
hälften. Vill du styra allt själv stänger du av *Scale line spacing with pen
width* under *Printer*.

Per penna kan du dessutom ställa:

* **Z offset** – kompenserar att pennor är olika långa,
* **Speed** – multiplicerar ritfarten (kulspetspennor vill ha lugnare tempo),
* **Tip** – vilken sorts spets (styr om Dwell dots fungerar),
* **Sharpen every** – meter innan den pennan behöver vässas.

## 6b. Live-styrning under ritning

I **Monitor** finns *LIVE CONTROL* som fungerar mitt i ett pågående jobb:

* **Speed** – matningsöverstyrning 25–300 % (`M220`). Slår igenom direkt.
* **Pen pressure** – höjer eller sänker ritnivån ±1,5 mm. Programmet skriver om
  Z-värdet i varje kommando innan det skickas, så du kan trimma pennanslaget
  medan den ritar.
* **Extra lift** – lägger till på pennlyftet om spetsen hakar i papperet.
* **Keep as default** skriver in de inställda värdena permanent.

## 6. Pauser

* **Pennbyte:** vid varje pennbyte parkeras huvudet, Z höjs och strömmen av
  kommandon stoppas. En dialog talar om vilken penna som ska i. Tryck
  **Continue** när du är klar.
* **Vässning:** slå på *Stop regularly to sharpen the pen* och ange intervall i
  meter (eller sätt värdet per penna). Perfekt för blyerts och stiftpennor.
* Pausen sköts från datorn och fungerar därför oavsett firmware. Kryssrutan
  *Also write M0 into the file* lägger dessutom in `M0` i filen så att pauserna
  fungerar även om du kör från SD-kort.

## 6c. Varför den är snabb

Mätt på ett foto, 900 px arbetsupplösning, 0,5 mm penna. **Pennlyften var
52–79 % av all tid** – inte ritandet. Tre åtgärder:

| Teknik | Före | Efter | |
|--------|------|-------|---|
| Crosshatch | 35,3 min | **10,0 min** | 72 % snabbare |
| Stippling | 133,2 min | **27,3 min** | 80 % snabbare |
| Flow field | 54,0 min | **9,2 min** | 83 % snabbare |
| Halftone | 64,3 min | **23,1 min** | 64 % snabbare |

1. **Connect strokes** (*Printer → Path optimisation*) – parallella skrafferings-
   drag slutar oftast en millimeter från där nästa börjar. Att rita rakt igenom
   är snabbare än att lyfta, och skarven hamnar i kanten av det skuggade
   området. 1 583 drag → 818 på ett foto; flow field 2 571 → 624. Stängs av
   automatiskt för tekniker där det skulle förstöra bilden (prickar, streck,
   konturer).
2. **Kort lyft för korta hopp** – ska pennan bara 2 mm åt sidan behöver den inte
   lyftas 2,5 mm. 0,6 mm räcker för att släppa papperet och går fem gånger
   fortare.
3. **Höj Z-taket** – Ender 3:ans firmware tillåter normalt bara 5 mm/s i Z, så
   att be om 900 mm/min gör ingen skillnad. Kryssa i *Raise the Z speed limit*
   så skickas `M203` vid start och återställs vid slutet. Ett tomt pennfäste är
   mycket lättare än ett hotend.

Programmet skickar också `M204` så maskinens acceleration matchar tidsestimatet,
och kan slå på `M420 S1` för att använda din bäddmesh – då blir pennanslaget
jämnt även på en skev bädd.

Tidsestimatet räknar numera med firmwarens verkliga Z-tak, så det stämmer.

## 6d. Testmönster – hitta rätt inställningar på fem minuter

Menyn **Tools** ritar fyra självdokumenterande ark. De går att spara och skicka
precis som vilken ritning som helst.

* **Pen height ladder** – samma streck på tretton olika pennhöjder, märkta
  −0,60 till +0,60 mm. Rita en gång, titta vilken rad som är skarp utan att
  gräva i papperet, skriv in den höjden. Tar 5 minuter i stället för en kväll av
  gissningar.
* **Speed ladder** – samma streck vid åtta matningshastigheter. En penna som
  hoppar vid 3000 mm/min är ofta perfekt vid 1200.
* **Pen test sheet** – fyllningar, skraffering, prickar och en spiral vid flera
  linjeavstånd, så du ser hur just din penna beter sig innan du binder upp en
  timme.
* **Registration sheet** – hörnkryss och en millimeterlinjal för att kontrollera
  skala och vinkelräthet.

Höjd- och hastighetsstegen fungerar genom att varje rad är ett eget lager med en
egen Z-förskjutning respektive matning inbakad i G-koden.

## 6e. SVG-export

**File → Export SVG…** skriver ritningen i millimeter med ett lager per penna,
rätt färg och rätt linjebredd, så den öppnas i verklig storlek i Inkscape och
Illustrator. Kontrollmätt: den exporterade geometrin stämmer med jobbet på
0,0003 %.

## 6f. Stenciler och skärverktyg

**Tools → Make stencils from this picture…** delar upp en bild i N stenciler så
att du kan spraya samma motiv i lager: ljusaste tonen genom ark 1, ark 2 ovanpå,
spraya igen – som screentryck.

Fönstret visar varje ark för sig och, med krysset i botten, **hur den färdiga
sprayningen skulle se ut**. Inställningarna som betyder mest:

| Inställning | Vad den gör |
|-------------|-------------|
| **Sheets** | Antal ark. Fler ark = mjukare toning, men fler sprayomgångar. |
| **Separate by** | *Tone levels* stackar gråtoner; *One sheet per paint colour* ger ett ark per färg i pennbiblioteket. |
| **Smallest feature** | Slivrar och öar tunnare än så här tas bort – de skulle ändå trilla ur. |
| **Bleed offset** | Krymper öppningen några tiondelar för att spray kryper in under kanten. |
| **Bridge width / spacing** | Bryggorna som håller ihop arket. |
| **Border** | Registreringsramen. |

**Bryggorna är hela poängen.** Insidan av ett "O" sitter inte fast i något och
faller ur så fort du skär runt det. Programmet letar upp varje sådan lös ö i
rastret och drar **tvärslåar** från ön ut till hållet material, och lägger
dessutom **flikar** (osågade bitar) längs konturerna så att spillbiten sitter
kvar. Flikarna placeras på raka partier, inte i hörn, och hålls borta från tunna
midjor. Varje ark **provskärs sedan i en simulering**: hela materialet rastreras,
snitten skärs, och om någon bit inte längre hänger ihop med ramen byggs arket om
med bredare tvärslåar. Ark som ändå inte går att rädda märks ut i klartext.

**Registrering:** ramen (*Border*, 6 mm som standard) skärs **identiskt på alla
ark** – lägg dem på varandra efter ramen så hamnar färgerna rätt. Kontrollmätt
genom hela kedjan: samma ram på varje ark inom 0,01 mm.

Trycker du **Create cut layers** hamnar arken som ett lager var i lagerlistan,
med bara det första påslaget – du skär ett kartongark i taget.

### Vad du skär med

*Pens*-panelen har **Add a cutting tool…** med fyra förinställningar:

| Verktyg | Hur den funkar |
|---------|----------------|
| **Drag knife** | Svängbart blad för vinyl och tunn kartong. Ett varv. |
| **Scalpel** | Fast blad, 3 varv, 0,25 mm djupare varje gång. |
| **Ordinary pen, scored N times** | *Inget blad alls* – en vanlig penna som går över samma linje 6 gånger tills papperet ger vika. |
| **Embossing / creasing tip** | Trycker en vikanvisning utan att bryta ytan. |

Det som faktiskt sker i G-koden:

* **Passes** – varje drag körs om N gånger, `Deeper each` mm djupare varje varv.
  På en sluten kontur behöver maskinen inte flytta sig alls mellan varven, och
  på en öppen linje går den bara fram och tillbaka. Ett sexvarvs skårjobb
  kostar alltså sex gånger ritsträckan men noll extra tomgång – och tidsestimatet
  räknar med det.
* **Blade offset** – ett svängblad släpar efter sin egen tapp med några tiondelar
  och hinner inte vända i ett hörn, så hörnen blir rundade. Programmet kör därför
  förbi hörnet exakt en bladoffset, **svänger runt hörnpunkten på en liten båge**
  medan spetsen står stilla, och fortsätter först då. Uppmätt mot en fysisk
  släpbladssimulering: 3 µm fel mot 121 µm utan kompensation.
* **Overcut** – en sluten kontur körs några millimeter förbi sin egen start,
  eftersom den biten skars medan bladet fortfarande vred sig.
* Skärlager **sys aldrig ihop** med grannstreck (det skulle bli ett snitt tvärs
  över stencilen) och märks med ✂ i pennlistan.

> Bladet skär på riktigt. Kör pennhöjdsguiden om igen när du bytt till blad –
> `Drawing Z` för ett blad är en annan höjd än för en penna, och för djupt
> betyder att du skär i skärmattan eller i bädden.

## 7. Bra att veta

* **Hastighet:** kulspets 1500–2500 mm/min, fineliner och blyerts tål mer.
  Ställs under *Printer → Speeds*.
* **Optimering:** programmet förenklar linjer, slår ihop drag som nästan möts
  och ordnar om ritordningen för minsta möjliga tomgång. På ett typiskt
  skrafferat foto blir tomgången ungefär 3 % av den ritade sträckan.
* **Estimat:** tiden räknas med accelerationen från *Printer* och stämmer
  normalt inom någon minut per timme.
* **Inställningar** sparas automatiskt i
  `~/Library/Application Support/PenPlotStudio/settings.json`.

## 7b. Andra skrivare än Ender 3

*Printer → Machine* har 19 färdiga profiler – Ender 3/5/7, CR-10, CR-6, Prusa
MK3S/MK4 och MINI, Artillery, Anycubic, Sovol, Elegoo Neptune 4, Voron 2.4 och
Trident, Duet, en ren GRBL-plotter, samt **Custom**. Profilen sätter bäddstorlek,
maxhöjd, parkeringspunkt, baudrate och firmware-familj.

Firmware-familjen avgör vilka valfria kommandon som får skickas, och det är inte
kosmetika:

| | Marlin | Klipper | RepRapFirmware | GRBL |
|---|---|---|---|---|
| Snabbare pennlyft | `M203 Z` i **mm/s** | hoppas över (Klipper har inte M203) | `M203 Z` i **mm/min** | hoppas över |
| Paus | `M0` | `PAUSE` | `M0` | – |
| Acceleration | `M204 P/T` | `M204 S` | `M204 P/T` | – |
| Bäddmesh | `M420 S1` | `BED_MESH_PROFILE LOAD=default` | `G29 S1` | – |
| Radnummer + checksumma | ja | ja | ja | nej |
| Statusrad `M117` | ja | ja | ja | nej |

Samma tal i `M203` betyder alltså **60 gånger** olika sak på Marlin och Duet –
skickas fel version tar ett jobb ett dygn i stället för en timme. Programmet
skriver rätt variant, och säger i G-koden när något hoppas över.

## 8. Om något strular

| Symptom | Trolig orsak |
|---------|--------------|
| Ingen port i listan | Fel USB-kabel (vissa är bara laddkablar), eller så behövs en CH340-drivrutin |
| Pennan ritar inte | `Drawing Z` för högt – kör pennhöjdsguiden igen |
| Pennan trycker för hårt | Höj `Drawing Z` med 0,1 mm i taget, eller använd fjädrande hållare |
| Streck där det ska vara tomt | `Lift` för liten, eller papperet buktar |
| Ritningen hamnar utanför bädden | Röd ram i förhandsvisningen – tryck **Fit to bed**. Programmet frågar också innan det skickar något som inte får plats |
| "The printer reset…" mitt i ett jobb | Strömglapp eller USB-glapp. Z-referensen är borta – kör pennhöjdsguiden igen |
| "No answer from the printer" | Vakthunden gav upp efter fyra försök. Kolla kabel, baudrate och om displayen väntar på ett knapptryck |
| Skrivaren gör ingenting efter Connect | Fel baudrate (Ender 3 kör oftast 115200) |

## 9. Projektet

```
penplot/
  core/          all logik, helt utan Qt-beroende i beräkningarna
    geometry.py    polylinjer: förenkling, hopslagning, ordning
    raster.py      bildinläsning och justeringar
    styles.py      kanter, konturer, skraffering, prickar, spiral, vågor
    separation.py  färgseparation till ett "bläcklager" per penna
    strokefont.py  inbyggt enlinjes-typsnitt (inkl. åäö)
    textsource.py  text → drag, både enlinje och TrueType-konturer
    pdfsource.py   PDF: vektorer + text, eller rendering av sidan
    pens.py        pennbiblioteket och skärverktygen
    techniques.py  registret med alla 21 rittekniker och deras parametrar
    stencil.py     bilden → N stencilark, med tvärslåar och flikar
    knife.py       bladoffset-kompensation och overcut
    profiles.py    19 skrivarprofiler och fyra firmware-familjer
    autotune.py    bildanalys + sluten loop som träffar måltiden
    pipeline.py    källa → teknik → placering → optimering (ett lager i taget)
    gcode.py       G-kod med pausmarkörer och flervarvsskärning
    printer.py     USB-streaming med checksumma och resend
  ui/            Qt-gränssnittet
    stencil_dialog.py  stencilfönstret
    panels/layers_panel.py  lagerlistan
tests/
  test_core.py         geometri, typsnitt, alla tekniker, pipeline, G-kod,
                       skärning (mot en fysisk släpbladssimulering) och stenciler
  test_stream.py       streaming mot sex simulerade firmware-beteenden
  test_interaction.py  dragning och skalning genom riktiga musevent
  render_ui.py         renderar fönstret till PNG utan skärm
  render_techniques.py kontaktark med alla tekniker på din egen bild
```

Kör testerna:

```bash
.venv/bin/python tests/test_core.py && .venv/bin/python tests/test_stream.py && .venv/bin/python tests/test_interaction.py
```

Packa om skrivbordsikonen (behövs bara om du flyttar projektmappen):

```bash
./package_app.sh
```
