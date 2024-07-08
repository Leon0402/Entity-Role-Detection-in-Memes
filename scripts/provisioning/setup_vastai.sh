# !/bin/bash

id=""
api_key=""

install_vastai() {
    pipx install vastai
    mkdir ./vastai
}

# Setup up api key to authenticate against vastai
api_key_helper() {

    if [ ! -f "./vastai/api_key.txt" ]; then
	echo ""
	echo "___! No API key present !___"
        echo ""
	echo "Copy from: https://cloud.vast.ai/account/"
	echo ""
	read -p "Please enter API key": api_key
	echo $api_key > ./vastai/api_key.txt
    else 
        echo "Found existing API key"
        api_key=$(cat ./vastai/api_key.txt)
    fi
}

# Setup up ssh key for vastai to create instances
ssh_key_helper() {
    
    if [ ! -f "./vastai/vastai_ssh" ]; then
        echo ""
        echo "___! No SSH key present !___"
        echo ""
        echo "Generating a new keypair..."
            ssh-keygen -f ./vastai/vastai_ssh -t rsa -N ''
        echo "...done"

        vastai create ssh-key "$(cat ./vastai/vastai_ssh.pub)"
    else
        echo "Found existing SSH key"
    fi
}


git_helper() {

    HOST_ADDRESS="$ip"
    CONFIG_LINES="Host $HOST_ADDRESS\n    ForwardAgent yes"
    if [ ! -f ~/.ssh/config ]; then
        touch ~/.ssh/config
    fi
    echo "backing up sshconfig"
    cp ~/.ssh/config ~/.ssh/config.bak

    eval "$(ssh-agent)"
    ssh-add -k
    ssh -T git@github.com
}

vastai_cli() {

    ### Select Instance
    echo ""	
    echo -e "\033[31mAvailable 4090s:\033[0m"
    echo ""
    offers=$(vastai search offers 'reliability > 0.99 num_gpus=1 gpu_name=RTX_4090 rented=False')
    echo "$offers"
    echo ""
    echo -e "\033[31mSelect ID of the instance to be provisioned.\033[0m"
    echo ""
    while true; do
        read -p "Enter ID: " id
        id=$(echo "$id" | xargs)
        break
        # if echo "$offers" | grep -Fq "$id"; then
        #     echo "Attempting to provision instance $id ..."
        #     break
        # else
        #         echo "Not a valid ID"
        # fi
    done

    ### Provision Instance
    vastai create instance "$id" --api-key "$api_key" --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel --disk 32 --ssh --direct --onstart scripts/provisioning/onstart_vastai.sh

    echo ""
    echo "Currently running instances:"
    echo ""

    instances=$(vastai show instances --api-key "$api_key")
    echo "$instances"
    id=$(echo "$instances" | sed -n '2s/ .*//p') 

    ip=$(curl --location --request GET 'https://console.vast.ai/api/v0/instances'        --header 'Accept: application/json'     --header "Authorization: Bearer $api_key" | grep -oP '"public_ipaddr": "\K[^"]+')

    echo "remote instance Ip is $ip"

}

connect_and_proxy() {

    echo "waiting for machine to provision (90s)"
    # sleep 90
    echo "Attempting to connect ..."
    echo "$id"
    echo "$api_key"
    ssh_string=$(vastai ssh-url "$id" --api-key "$api_key" | cut -c 7-)
    echo "$ssh_string"
    user_host=$(echo "$ssh_string" | cut -d':' -f1)
    echo "$user_host"
    port=$(echo "$ssh_string" | cut -d':' -f2)
    echo "$port"
    ssh -v -p $port $user_host -L 8080:localhost:8080 -A -i ./vastai/vastai_ssh -o "ConnectTimeout 3" -o "StrictHostKeyChecking no" -o "UserKnownHostsFile /dev/null" "$@"
    ls -la
}


install_vastai
api_key_helper
ssh_key_helper

vastai_cli
git_helper
connect_and_proxy