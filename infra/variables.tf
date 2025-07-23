variable "resource_group_name" {
  default = "mlp-inference-rg"
}

variable "location" {
  default = "westeurope"
}

variable "app_service_plan_name" {
  default = "mlp-inference-plan"
}

variable "app_service_name" {
  default = "mlp-inference-api"
}