function map = qTable2Map(qTable)

global Lt;
if mod(qTable,Lt) == 0
    map=[qTable/Lt,Lt];
else
    map=[floor(qTable/Lt)+1,mod(qTable,Lt)];
end

