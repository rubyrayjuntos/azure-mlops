resource "azurerm_search_service" "rag" {
  name                         = "srch${var.prefix}${var.postfix}${var.env}"
  resource_group_name          = var.rg_name
  location                     = var.location
  sku                          = var.sku
  replica_count                = var.replica_count
  partition_count              = var.partition_count
  public_network_access_enabled = var.public_network_access_enabled
  local_authentication_enabled = false

  tags = var.tags
}
