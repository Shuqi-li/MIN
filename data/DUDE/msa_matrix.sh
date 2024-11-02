for file in /home/shuqi_li/drugVQA-master/data/DUDE/msa/*
do

file_name=${file##*/}
file_name=${file_name%%.*}
out_path=/home/shuqi_li/drugVQA-master/data/DUDE/msa_fasta/"${file_name}".fasta
reformat.pl $file ${out_path} -M first -r

conserv_score=$(Rscript /home/shuqi_li/drugVQA-master/data/DUDE/generate_conserv_score.r "${out_path}")
echo $(cat /home/shuqi_li/drugVQA-master/data/DUDE/tmpt_identity.txt) >> /home/shuqi_li/drugVQA-master/data/DUDE/msa_score/"${file_name}".fasta

#将fasta的msa转换为i矩阵格式
#首先将文件中每个以>开始的行替换为~~
sed -i "/^>/c~~" $out_path
#删除所有的换行符
sed -i ':t;N;s/\n//;b t' $out_path
#将~~替换为换行符
sed -i 's/~~/\n/g' $out_path
#删除第一空行
sed -i '1d' $out_path
#将两个文件合并
cat /home/shuqi_li/drugVQA-master/data/DUDE/msa_score/"${file_name}".fasta $out_path > /home/shuqi_li/drugVQA-master/data/DUDE/msa_matrix/"${file_name}".fasta

done
