# Approche TP1 / TP2 :

**TP1 :** 
Le but était de trouver la tendance d'un jeu de données afin de pouvoir produire une attente sur une nouvelle donnée.

**TP2 :** 
Le but était de comprendre les caractéristiques des données afin de pouvoir comprendre en quoi elles peuvent être distinctes et donc classées. Ensuite, on a la possiblité de classer une nouvelle donnée inconnue en fonction de ses caractéristiques, et de celles des données d'entrainements.

# Principe de régularisation

**Régularisation :**
Principe permettant d'accorder plus ou moins d'importance aux caractéristiques parfois particulières des données d'entarinements. Avec un lambda nul, on colle le plus possible aux données d'entrainements, mais on tient aussi compte des erreurs potentielles qu'elles comportent. 

# Influence du lambda

**Lambda :** Avec un lambda très grand, on ne retient d'une tendance générale des données d'entrainements, mais globalement la plupart des données d'entrainements n'appartiennent plus à la bonne classe, car elles contiennent un certain biai. On peut ajouter que le lambda tend à lisser la frontière entre les deux classes, et qu'a faible lambda la frontère est plus bruitée.