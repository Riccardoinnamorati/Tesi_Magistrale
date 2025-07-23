# FALKO Plotter

Il plotter FALKO è un nodo ROS2 che visualizza in tempo reale:
1. I percorsi dei robot nello spazio
2. I match tra keypoint quando i robot si riconoscono

**Caratteristiche principali**: Il plotter mostra un plot interattivo che si aggiorna in tempo reale, simile al keypoint_detector.

## Caratteristiche

- **Visualizzazione trajettorie**: Mostra i percorsi di tutti i robot con colori diversi
- **Evidenziazione match**: Quando due robot matchano le scansioni, vengono evidenziati in rosso
- **Integrazione velocità**: Se non riceve dati di posizione, integra le velocità dalle posizioni iniziali
- **Compatibilità Webots**: Supporta topic odometry e pose da Webots
- **Aggiornamento live**: Plot interattivo che si aggiorna ogni 0.1 secondi

## Visualizzazione

Il plotter mostra una finestra con due subplot:

- **Plot sinistro**: Trajectorie dei robot
  - Linee colorate per ogni robot
  - Punti colorati per posizioni correnti
  - Punti rossi ingranditi per robot che hanno appena matchato (ultimi 3 secondi)

- **Plot destro**: Visualizzazione keypoint matches
  - Keypoint del primo robot (cerchi) e del secondo robot (triangoli) 
  - Linee tratteggiate che collegano i keypoint matchati
  - Numerazione dei match nel punto medio delle connessioni
  - Colori coordinati con le trajectorie dei robot
  - Informazioni sul match (ID robot e numero di coppie)

## Dipendenze

- matplotlib >= 3.3.0
- numpy >= 1.19.0
- rclpy
- geometry_msgs
- nav_msgs
- choirbot_interfaces
- tf2_msgs
