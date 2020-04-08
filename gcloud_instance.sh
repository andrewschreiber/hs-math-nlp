#! /bin/bash

# Not meant to be run as a bash script, yet.


# XXX First, create a disk  - DOESNT WORK WITH SPOT INSTANCES CAUSE OF GROUP
# gcloud beta compute disks create hs-math-ssd --project=hs-math-nlp --type=pd-ssd --size=60GB --zone=us-west1-a --physical-block-size=4096 --image=c2-deeplearning-pytorch-1-3-cu100-20191219

# Create a cloud storage bucket
# https://console.cloud.google.com/storage/browser
# Upload from https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz


# XXX Second, create a cheap instance to download your data and code
# gcloud beta compute --project=hs-math-nlp instances create instance-1 --zone=us-west1-a --machine-type=n1-standard-1 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=8190450584-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --disk=name=pytorch-math-ssd,device-name=pytorch-math-ssd,mode=rw,boot=yes --reservation-affinity=any


export IMAGE_FAMILY="pytorch-latest-gpu" \
export ZONE="us-west1-b" \
export INSTANCE_NAME="my-fastai-instance-e" \
export INSTANCE_TYPE="n1-highmem-8" \
export PROJECT="hs-math-nlp"


# Dummy instance
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible \
        --scopes storage-rw \
        --metadata-from-file startup-script=gce/startup.sh 


# Tail logs
watch -n 2 "gcloud compute --project=$PROJECT instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE | tail -40"        



export INSTANCE_NAME="my-fastai-instance-l" && \
gcloud compute instances create $INSTANCE_NAME \
        --zone=$ZONE \
        --image-family=$IMAGE_FAMILY \
        --image-project=deeplearning-platform-release \
        --maintenance-policy=TERMINATE \
        --machine-type=$INSTANCE_TYPE \
        --boot-disk-size=200GB \
        --metadata="install-nvidia-driver=True" \
        --preemptible \
        --scopes storage-rw \
        --metadata-from-file startup-script=gce/prestartup.sh \
&& watch -n 2 "gcloud compute --project=$PROJECT instances get-serial-port-output $INSTANCE_NAME --zone=$ZONE | tail -40"        





# Create your preemptible instance template
gcloud beta compute --project=hs-math-nlp instance-templates create template-math 
  --machine-type=n1-standard-8 --network=projects/hs-math-nlp/global/networks/default --network-tier=PREMIUM --metadata=IS_SPOT=true --no-restart-on-failure --maintenance-policy=TERMINATE --preemptible --service-account=8190450584-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_write,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=type=nvidia-tesla-v100,count=8, mode=rw,boot=yes --reservation-affinity=any


# Second, create a instance group to autoscale back to 1. Starts an instance!
gcloud compute --project=hs-math-nlp instance-groups managed create instance-group-pre-v100 --base-instance-name=instance-group-pre-v100 --template=template-pre-v100 --size=1 --zone=us-west1-b

gcloud beta compute --project "hs-math-nlp" instance-groups managed set-autoscaling "instance-group-pre-v100" --zone "us-west1-b" --cool-down-period "15" --max-num-replicas "1" --min-num-replicas "1" --target-cpu-utilization "0.6" --mode "on"


# SSH in
gcloud beta compute --project "hs-math-nlp" ssh --zone "us-west1-b" "instance-group-pre-v100-bfwl"


# wget https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz