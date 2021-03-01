net=$Dataria/Models/RAFT/models/raft-sintel.pth


for in_data in  /media/etienne/Lacie/Inria//ChAOS_datasets/ChAOS_faster/scenes/*/Renders/*
do
  out_data=$(echo $in_data | sed "s/scenes/OpticalFlow/g")
  echo out_data : $out_data

  python3 demo.py --model $net\
  		--path $in_data\
  		--save\
  		--path_save $out_data
done
