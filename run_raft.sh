in_data=$Dataria/ChAOS_datasets/ChAOS_faster/scenes//Renders/camera1/
out_data=$Dataria/ChAOS_datasets/ChAOS_faster/OpticalFlow/
net=$Dataria/Models/RAFT/models/raft-sintel.pth

python3 demo.py --model $net\
		--path $in_data\
		--save\
		--path_save $out_data
