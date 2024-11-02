# 提取contactmap中的第二行seq，生成msa文件，并以contactmap第一行命名
for file in /home/shuqi_li/drugVQA-master/data/DUDE/contactMap/*
do
query=$(sed -n '2p' $file)
name=$(sed -n '1p' $file)
echo ">${name}" >> /home/shuqi_li/drugVQA-master/data/DUDE/protein_seq/"$name".fas
echo $query >> /home/shuqi_li/drugVQA-master/data/DUDE/protein_seq/"$name".fas

hhblits -i /home/shuqi_li/drugVQA-master/data/DUDE/protein_seq/"${name}".fas -o /home/shuqi_li/drugVQA-master/data/DUDE/msa_uniref_hhr/"${name}".hhr -oa3m /home/shuqi_li/drugVQA-master/data/DUDE/msa_uniref/"${name}".a3m -n 1 -e 0.000001 -d /home/shuqi_li/uniref30/UniRef30_2020_06

done

