kubectl apply -f book.yaml
sleep 5
kubectl exec maciej-devel mkdir /home/user/.ssh
kubectl cp ~/.ssh/id_rsa.pub maciej-devel:/home/user/.ssh/authorized_keys
kubectl port-forward maciej-devel 2222:22 8888:8888
kubectl exec maciej-devel sudo service ssh restart
