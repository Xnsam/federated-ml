# federated-ml
Examples, demos and projects implementing federated machine learning

-----
Arch
-----

![alt text](images/image.png)



-----
How to run in cli
-----
<code>
terraform init <br/>
![alt text](images/t_init.png)
terraform plan <br/>
![alt text](images/t_plan.png)
![alt text](images/t_plan2.png)
terraform apply<br/>
![alt text](images/t_apply.png)
</code>

If any updates are made to `main.tf`

then rerun all steps
<code>
terraform init -upgrade <br>
terraform plan <br>
terraform apply <br>
</code>

Any updates to the code files

then only run 
`terraform apply -replace="docker_image.fed_app`
this replaces the build 

finally, `docker ps` should result <br>
![alt text](images/d_ps.png)

Now, to stop everything <br>
`terraform destory`