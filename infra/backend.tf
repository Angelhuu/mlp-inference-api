terraform {
  backend "azurerm" {
    resource_group_name   = "rg-terraform-state"
    storage_account_name  = "sttfstate24"
    container_name        = "tfstate"
    key                   = "projetdevops.tfstate"
  }
}