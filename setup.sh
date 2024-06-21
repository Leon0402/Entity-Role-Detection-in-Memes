# !/bin/bash

id=""
api_key=""

ssh_key_helper() {
    
    if [ ! -f "vastai_ssh" ]; then
	echo ""
	echo "___! No SSH key present !___"
	echo ""
	echo "Generating a new keypair..."
        ssh-keygen -f ./vastai_ssh -t rsa -N ''
	echo "...done"
	echo "Add the public key:"
	echo ""
	cat ./vastai_ssh.pub
	echo ""
	echo "to https://cloud.vast.ai/account/"
	echo "Waiting. Press any key to continue"
	read -n 1 -s
    else
	echo "Found existing SSH key"
	ssh_key_prv=$(cat vastai_ssh)
	ssh_key_pub=$(cat vastai_ssh.pub)
    fi
}

api_key_helper() {

    if [ ! -f "api_key.txt" ]; then
	echo ""
	echo "___! No API key present !___"
        echo ""
	echo "Copy from: https://cloud.vast.ai/account/"
	echo ""
	read -p "Please enter API key": api_key
	echo $api_key > api_key.txt
    else 
        echo "Found existing API key"
        api_key=$(cat api_key.txt)
    fi
}

cli_setup() {
    python3 -m venv vastai-helper-venv
    source ./vastai-helper-venv/bin/activate
    pip install vastai
}

git_helper() {

    HOST_ADDRESS="$ip"
    CONFIG_LINES="Host $HOST_ADDRESS\n    ForwardAgent yes"
    if [ ! -f ~/.ssh/config ]; then
        touch ~/.ssh/config
    fi
    echo "backing up sshconfig"
    cp ~/.ssh/config ~/.ssh/config.bak

    # Append the lines to the config file if they are not already present
    if ! grep -q "Host $HOST_ADDRESS" ~/.ssh/config; then
        echo -e "$CONFIG_LINES" >> ~/.ssh/config
        echo "Configuration added successfully."
    else
        echo "Configuration for Host $HOST_ADDRESS already exists."
    fi
    eval "$(ssh-agent)"
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
	if echo "$offers" | grep -Fq "$id"; then
	    echo "Attempting to provision instance $id ..."
	    break
	else
            echo "Not a valid ID"
	fi
    done

    ### Provision Instance

    vastai create instance "$id" --api-key "$api_key" --image pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel --disk 32 --onstart ./onstart.sh

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
    sleep 90
    echo "Attempting to connect ..."
    echo "$id"
    echo "$api_key"
    ssh_string=$(vastai ssh-url "$id" --api-key "$api_key" | cut -c 7-)
    echo "$ssh_string"
    user_host=$(echo "$ssh_string" | cut -d':' -f1)
    echo "$user_host"
    port=$(echo "$ssh_string" | cut -d':' -f2)
    echo "$port"
    ssh -v -p $port $user_host -L 8080:localhost:8080 -i ./vastai_ssh -o "ConnectTimeout 3" -o "StrictHostKeyChecking no" -o "UserKnownHostsFile /dev/null" "$@"
    ls -la
}

ssh_key_helper
api_key_helper
cli_setup
vastai_cli
git_helper
#connect_and_proxy
