terraform {
  required_providers {
    docker = {
      source = "kreuzwerker/docker"
      version = ">= 3.0.12"
    }
  }
}

provider "docker" {

}


# create a dedicated network for fed cluster
resource "docker_network" "fed_network" {
  name = "federated_ml_network"
}

# build multi-stage docker image
resource "docker_image" "fed_app" {
  name = "fed-ml-app:latest"
  build {
    context = "." # path to your dockerfile and source code
  }
}

resource "docker_volume" "server_data" {
  name = "fed_server_volume"
}

# the aggregator server
resource "docker_container" "server" {
  name = "fed_server"
  image = docker_image.fed_app.image_id

  volumes {
    volume_name = docker_volume.server_data.name
    container_path = "/app/data" # maps to path used in python
  }

  networks_advanced {
    name = docker_network.fed_network.name
  }
  command = ["python", "server.py"]
  ports {
    internal = 8000
    external = 8000
  }
}

# the edge clients (iterative creation)
resource "docker_container" "clients" {
  count = 2 # change this to increase number of clients
  name = "client-${count.index + 1}"
  image = docker_image.fed_app.image_id

  # resource constraints
  # limit memory
  memory = 1024

  # limit cpu: 1.0 represent on full core of your host
  # most PIs have 4 cores but slower than pc, so using 0.5 or 1 gives realistic feel
  cpu_set = "0"
  cpu_shares = 1024

  networks_advanced {
    name = docker_network.fed_network.name
  }

  command = ["python", "client.py"]

  env = [
    "PYTHONBUFFERED=1",
    "APP_ROLE=client",
    "MODEL_TYPE=jax_light",
    "CLIENT_ID=pi-${count.index + 1}",
    "SERVER_URL=http://fed_server:8000", 
    "XLA_PYTHON_CLIENT_PREALLOCATE=false",
    "XLA_PYTHON_CLIENT_MEM_FRACTION=.50"
  ]

  # ensure the server is up to before clients start
  depends_on = [docker_container.server]
  
  # mount the local sensor data
  volumes {
    host_path = abspath("${path.module}/client_data_${count.index}")
    container_path = "/app/data"
  }
}