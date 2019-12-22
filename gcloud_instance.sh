#! /bin/bash

gcloud beta compute --project=hs-math-nlp instance-templates create spot-template 
  --machine-type=n1-standard-8 --network=projects/hs-math-nlp/global/networks/default --network-tier=PREMIUM --metadata=IS_SPOT=true --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible --service-account=8190450584-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --image=c2-deeplearning-pytorch-1-3-cu100-20191112 --image-project=ml-images --boot-disk-size=80GB --no-boot-disk-auto-delete --boot-disk-type=pd-ssd --boot-disk-device-name=spot-template --reservation-affinity=any

gcloud compute --project=hs-math-nlp instance-groups managed create spot-instance-group 
  --base-instance-name=spot-instance-group --template=spot-template --size=1 --zone=us-central1-a

gcloud beta compute --project "hs-math-nlp" instance-groups managed set-autoscaling "spot-instance-group" 
  --zone "us-central1-a" --cool-down-period "60" --max-num-replicas "1" --min-num-replicas "1" --target-cpu-utilization "1" --mode "on"


gcloud beta compute --project=hs-math-nlp instance-templates create spot-template 
  --machine-type=n1-standard-8 --network=projects/hs-math-nlp/global/networks/default --network-tier=PREMIUM --metadata=IS_SPOT=true --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible --service-account=8190450584-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=type=nvidia-tesla-v100,count=8 --image=c2-deeplearning-pytorch-1-3-cu100-20191112 --image-project=ml-images --boot-disk-size=80GB --no-boot-disk-auto-delete --boot-disk-type=pd-ssd --boot-disk-device-name=spot-template --reservation-affinity=any
