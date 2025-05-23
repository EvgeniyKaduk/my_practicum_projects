# Отбор коров для экофермы
## Описание проекта
Мы работаем в IT-компании, которая выполняет на заказ проекты по машинному обучению. К нам обратился фермер, владелец молочного хозяйства «Вольный луг». Он хочет купить бурёнок, чтобы расширить поголовье стада коров. Для этого он заключил выгодный контракт с ассоциацией пастбищ «ЭкоФерма». Условия позволяют фермеру очень тщательно отобрать коров. Фермер хочет, чтобы каждая бурёнка давала не менее 6000 килограммов молока в год, а надой был вкусным — строго по его критериям, ничуть не хуже.

Поэтому он просит нас разработать модель машинного обучения, которая поможет ему управлять рисками и принимать объективное решение о покупке. «ЭкоФерма» готова предоставить подробные данные о своих коровах. Нам нужно создать две прогнозные модели для отбора бурёнок в поголовье:

Первая будет прогнозировать возможный удой у коровы (целевой признак Удой);
Вторая — рассчитывать вероятность получить вкусное молоко от коровы (целевой признак Вкус молока).

С помощью модели нужно отобрать коров по двум критериям:
  - Средний удой за год — не менее 6000 килограммов,
  - Молоко должно быть вкусным.

## Описание данных
Заданы следующие признаки:
  - ЭКЕ (Энергетическая кормовая единица) — измерение питательности корма коровы;
  - сырой протеин — содержание сырого протеина в корме, в граммах;
  - СПО (Сахаро-протеиновое соотношение) — отношение сахара к протеину в корме коровы;
  - тип пастбища — ландшафт лугов, на которых паслась корова;
  - номер коровы;
  - порода коровы;
  - возраст (менее 2 лет, более 2 лет);
  - порода папы коровы;
  - имя папы-быка
  - содержание жиров в молоке, в процентах;
  - содержание белков в молоке, в процентах;
  - оценка вкуса молока по личным критериям фермера;
  - масса молока, которую корова даёт в год, в килограммах.

Параметры корма ЭКЕ, Сырой протеин, г и СПО отсутствуют для коров, которых фермер хочет изучить перед покупкой. Технологи заказчика пересмотрели подход к кормлению: для новых коров планируется увеличить значения каждого из этих параметров на 5%.

## Используемые библиотеки
*pandas*, *matplotlib*, *numpy*, *seaborn*, *os*, *math*, *scikit-learn*

## Результат
Построено две модели: линейная регресия для прогноза удоев молока и логистическая регрессия для прогноза вкуса. В результате наложения двух моделей из 20 коров, предложенных к покупке, было выбрано 7 коров, удовлетворяющих критериям 
