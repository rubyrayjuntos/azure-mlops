variable "rg_name" {
  type        = string
  description = "Resource group name"
}

variable "location" {
  type        = string
  description = "Azure region"
}

variable "prefix" {
  type        = string
  description = "Resource prefix"
}

variable "postfix" {
  type        = string
  description = "Resource postfix"
}

variable "env" {
  type        = string
  description = "Environment name"
}

variable "sku" {
  type        = string
  description = "Azure AI Search SKU"
  default     = "basic"
}

variable "replica_count" {
  type        = number
  description = "Replica count"
  default     = 1
}

variable "partition_count" {
  type        = number
  description = "Partition count"
  default     = 1
}

variable "public_network_access_enabled" {
  type        = bool
  description = "Whether public network access is allowed"
  default     = true
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Tags to apply to the search service"
}
