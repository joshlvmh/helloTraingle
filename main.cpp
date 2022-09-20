// replaces include <vulkan/vulkan.h> for window management, automatically loads Vulkan header
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

// linear algebra
// #define GLM_FORCE_RADIANS
// #define GLM_FORCE_DEPTH_ZERO_TO_ONE
// #include <glm/vec4.hpp>
// #include <glm/mat4x4.hpp>

// error handling
#include <iostream>
#include <stdexcept>
// EXIT_SUCCESS & EXIT_FAILURE macros
#include <cstdlib>

#include <vector>
#include <algorithm>
#include <cstring>
#include <optional>
#include <set>
#include <limits>
#include <fstream>
#include <array>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

// allow rendering of one frame to not interfere with the recording of the next
const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> validationLayers = {
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
	VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

// proxy function to lookup vkCreateDebugUtilsMessengerEXT function address
// (not loaded automatically as extension function)
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr) {
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	} else {
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

// proxy function to destroy debug messenger
// should be static class function or function outside of class
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr) {
		func(instance, debugMessenger, pAllocator);
	}
}

struct QueueFamilyIndices {
	/* C++17 <optional> data structure to distinguish between value existing or not,
	 * as value here could be anything, including 0, and be a valid graphicsFamily
	 *
	 * std::cout << std::boolalpha << graphicsFamily.has_value() << std::endl; // false
	 * graphicsFamily = 0;
	 * std::cout << std::boolalpha << graphicsFamily.has_value() << std::endl; // true
	 **/
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	bool isComplete() {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct SwapChainSupportDetails {
	// basic surface capabilities (min/max number of images in swap chain, min/max width/height of images)
	VkSurfaceCapabilitiesKHR capabilities;
	// surfaace formats (pixel format, colour space)
	std::vector<VkSurfaceFormatKHR> formats;
	// available present modes
	std::vector<VkPresentModeKHR> presentModes;
};

struct Vertex {
	glm::vec2 pos;
	glm::vec3 color;

	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Vertex);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Vertex, pos);
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Vertex, color);

		return attributeDescriptions;
	}
};

const std::vector<Vertex> vertices = {
	{{0.0f, -0.5f}, {1.0f, 1.0f, 1.0f}},
	{{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
	{{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
};

class HelloTriangleApplication {
	public:
		void run() {
			initWindow();
			initVulkan();
			mainLoop();
			cleanup();
		}

	private:
		// window reference
		GLFWwindow* window;
		// handle to instance
		VkInstance instance;
		VkDebugUtilsMessengerEXT debugMessenger;
		VkSurfaceKHR surface;
		VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
		// logical device
		VkDevice device;
		// queue handles
		VkQueue graphicsQueue;
		VkQueue presentQueue;
		
		VkSwapchainKHR swapChain;
		std::vector<VkImage> swapChainImages;
		VkFormat swapChainImageFormat;
		VkExtent2D swapChainExtent;

		std::vector<VkImageView> swapChainImageViews;

		VkRenderPass renderPass;
		VkPipelineLayout pipelineLayout;
		VkPipeline graphicsPipeline;

		std::vector<VkFramebuffer> swapChainFramebuffers;
		VkCommandPool commandPool;
		// one for each frame in flight
		std::vector<VkCommandBuffer> commandBuffers;
		std::vector<VkSemaphore> imageAvailableSemaphores;
		std::vector<VkSemaphore> renderFinishedSemaphores;
		std::vector<VkFence> inFlightFences;
		uint32_t currentFrame = 0;
		bool framebufferResized = false;

		VkBuffer vertexBuffer;
		VkDeviceMemory vertexBufferMemory;

		// initialise GLFW and create a window
		void initWindow() {
			glfwInit();

			// do not create OpenGL context (default behaviour)
			glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
			// disable window resizing
		//	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

			// create window (width, height, title, specify monitor [opt], openGL specific param)
			window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
			glfwSetWindowUserPointer(window, this);
			glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		}

		static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
			auto app = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
			app->framebufferResized = true;
		}

		void initVulkan() {
			createInstance();
			setupDebugMessenger();
			createSurface();
			pickPhysicalDevice();
			createLogicalDevice();
			createSwapChain();
			createImageViews();
			createRenderPass();
			createGraphicsPipeline();
			createFramebuffers();
			createCommandPool();
			createVertexBuffer();
			createCommandBuffers();
			createSyncObjects();
		}

		void mainLoop() {
			// run until either error occurs or window is closed
			while (!glfwWindowShouldClose(window)) {
				glfwPollEvents();
				drawFrame();
			}

			vkDeviceWaitIdle(device);
		}

		void cleanup() {
			cleanupSwapChain();

			vkDestroyBuffer(device, vertexBuffer, nullptr);
			vkFreeMemory(device, vertexBufferMemory, nullptr);

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
				vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
				vkDestroyFence(device, inFlightFences[i], nullptr);
			}

			vkDestroyCommandPool(device, commandPool, nullptr);
			vkDestroyPipeline(device, graphicsPipeline, nullptr);
			vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			vkDestroyRenderPass(device, renderPass, nullptr);

			// instance not a parameter as logical devices do not interact directly with instances
			vkDestroyDevice(device, nullptr);

			if (enableValidationLayers) {
				DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
			}

			vkDestroySurfaceKHR(instance, surface, nullptr);
			// (handle, optional allocator callback)
			vkDestroyInstance(instance, nullptr);
			
			glfwDestroyWindow(window);

			glfwTerminate();
		}

		void cleanupSwapChain() {
			for (auto framebuffer : swapChainFramebuffers) {
				vkDestroyFramebuffer(device, framebuffer, nullptr);
			}
			for (auto imageView : swapChainImageViews) {
				vkDestroyImageView(device, imageView, nullptr);
			}
			vkDestroySwapchainKHR(device, swapChain, nullptr);

		}

		void recreateSwapChain() {
			int width = 0, height = 0;
			glfwGetFramebufferSize(window, &width, &height);
			// minimized window size = 0
			while (width == 0 || height == 0) {
				glfwGetFramebufferSize(window, &width, &height);
				glfwWaitEvents();
			}
				
			vkDeviceWaitIdle(device);

			cleanupSwapChain();

			createSwapChain();
			createImageViews();
			createFramebuffers();
		}

		void createVertexBuffer() {
			VkBufferCreateInfo bufferInfo{};
			bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
			bufferInfo.size = sizeof(vertices[0]) * vertices.size();
			bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
			bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

			if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
				throw std::runtime_error("failed to create vertex buffer.");
			}

			VkMemoryRequirements memRequirements;
			vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

			VkMemoryAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
			allocInfo.allocationSize = memRequirements.size;
			allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

			if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate vertex buffer memory.");
			}

			vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

			void* data;
			vkMapMemory(device, vertexBufferMemory, 0, bufferInfo.size, 0, &data);
			memcpy(data, vertices.data(), (size_t) bufferInfo.size);
			vkUnmapMemory(device, vertexBufferMemory);
		}

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
			VkPhysicalDeviceMemoryProperties memProperties;
			vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

			for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
				if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
					return i;
				}
			}
			throw std::runtime_error("failed to find suitable memory type.");
		}

		void drawFrame() {
			/* rendering frame steps:
			 * 	- wait for previous frame to finish
			 * 	- acquire image from the swap chain
			 * 	- record a command buffer which draws the scene onto that image
			 * 	- submit the recorded command buffer
			 * 	- present the swap chain image
			 */

			// wait for previous frame
			vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

			// acqure image from swap chain
			uint32_t imageIndex;
			VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
			if (result == VK_ERROR_OUT_OF_DATE_KHR) {
				recreateSwapChain();
				return;
			} else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
				throw std::runtime_error("failed to acquire swap chain image.");
			}
			// reset fence after waiting & confirming carrying on
			vkResetFences(device, 1, &inFlightFences[currentFrame]);

			// record a command buffer
			vkResetCommandBuffer(commandBuffers[currentFrame], 0);
			recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

			// submit the command buffer
			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
			VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
			submitInfo.waitSemaphoreCount = 1;
			submitInfo.pWaitSemaphores = waitSemaphores;
			submitInfo.pWaitDstStageMask = waitStages;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
			VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
			submitInfo.signalSemaphoreCount = 1;
			submitInfo.pSignalSemaphores = signalSemaphores;

			if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
				throw std::runtime_error("failed to submit draw command buffer.");
			}

			// present the swap chain image
			VkPresentInfoKHR presentInfo{};
			presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
			presentInfo.waitSemaphoreCount = 1;
			presentInfo.pWaitSemaphores = signalSemaphores;

			VkSwapchainKHR swapChains[] = {swapChain};
			presentInfo.swapchainCount = 1;
			presentInfo.pSwapchains = swapChains;
			presentInfo.pImageIndices = &imageIndex;
			presentInfo.pResults = nullptr;

			result = vkQueuePresentKHR(presentQueue, &presentInfo);

			if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
				framebufferResized = false;
				recreateSwapChain();
			} else if (result != VK_SUCCESS) {
				throw std::runtime_error("failed to present swap chain image.");
			}

			currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
		}

		void createSyncObjects() {
			imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
			renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
			inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

			VkSemaphoreCreateInfo semaphoreInfo{};
			semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

			VkFenceCreateInfo fenceInfo{};
			fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
			fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

			for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
				if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
				    vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
				    vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create sync objects for a frame.");
				}
			}
		}

		void createCommandBuffers() {
			commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

			VkCommandBufferAllocateInfo allocInfo{};
			allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
			allocInfo.commandPool = commandPool;
			allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			allocInfo.commandBufferCount = (uint32_t) commandBuffers.size();

			if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
				throw std::runtime_error("failed to allocate command buffers.");
			}
		}

		void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = 0;
			beginInfo.pInheritanceInfo = nullptr;

			if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
				throw std::runtime_error("failed to begin recording command buffer.");
			}

			VkRenderPassBeginInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			renderPassInfo.renderPass = renderPass;
			renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
			renderPassInfo.renderArea.offset = {0, 0};
			renderPassInfo.renderArea.extent = swapChainExtent;
			// clear here is black with 100% opacity
			VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
			renderPassInfo.clearValueCount = 1;
			renderPassInfo.pClearValues = &clearColor;
			// vkCmd prefix : functions that record commands
			// all return void so no error handling until finish recording
			vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
			// bind the graphics pipeline (2nd argument specifies graphics or compute pipeline)
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
			
			VkBuffer vertexBuffers[] = {vertexBuffer};
			VkDeviceSize offsets[] = {0};
			vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

			// as scissor and viewport dynamic, set here
			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = static_cast<float>(swapChainExtent.width);
			viewport.height = static_cast<float>(swapChainExtent.height);
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
			VkRect2D scissor{};
			scissor.offset = {0, 0};
			scissor.extent = swapChainExtent;
			vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

			// draw command for triangle (buffer, vertexCount, instanceCount, firstVertex, firstInstance)
			vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertices.size()), 1, 0, 0);

			vkCmdEndRenderPass(commandBuffer);

			if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
				throw std::runtime_error("failed to record command buffer.");
			}
		}

		void createCommandPool() {
			QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

			VkCommandPoolCreateInfo poolInfo{};
			poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
			poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			poolInfo.queueFamilyIndex= queueFamilyIndices.graphicsFamily.value();

			if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
				throw std::runtime_error("failed to create command pool.");
			}
		}

		void createFramebuffers() {
			swapChainFramebuffers.resize(swapChainImageViews.size());

			for (size_t i = 0; i < swapChainImageViews.size(); i++) {
				VkImageView attachments[] = {
					swapChainImageViews[i]
				};

				VkFramebufferCreateInfo framebufferInfo{};
				framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
				framebufferInfo.renderPass = renderPass;
				framebufferInfo.attachmentCount = 1;
				framebufferInfo.pAttachments = attachments;
				framebufferInfo.width = swapChainExtent.width;
				framebufferInfo.height = swapChainExtent.height;
				framebufferInfo.layers = 1;

				if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create framebuffer.");
				}
			}
		}

		void createRenderPass() {
			VkAttachmentDescription colorAttachment{};
			colorAttachment.format = swapChainImageFormat;
			colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
			colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
			colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
			colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
			colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
			colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
			colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

			VkAttachmentReference colorAttachmentRef{};
			// index 0 in array of VkAttachmentDescription [only a single here]
			colorAttachmentRef.attachment = 0;
			colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

			VkSubpassDescription subpass{};
			subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
			subpass.colorAttachmentCount = 1;
			subpass.pColorAttachments = &colorAttachmentRef;

			VkSubpassDependency dependency{};
			dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
			dependency.dstSubpass = 0;
			dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.srcAccessMask = 0;
			dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
			dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

			VkRenderPassCreateInfo renderPassInfo{};
			renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
			renderPassInfo.attachmentCount = 1;
			renderPassInfo.pAttachments = &colorAttachment;
			renderPassInfo.subpassCount = 1;
			renderPassInfo.pSubpasses = &subpass;
			renderPassInfo.dependencyCount = 1;
			renderPassInfo.pDependencies = &dependency;

			if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
				throw std::runtime_error("failed to create render pass.");
			}

		}	
		
		void createGraphicsPipeline() {
			auto vertShaderCode = readFile("shaders/vert.spv");
			auto fragShaderCode = readFile("shaders/frag.spv");

			VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
			VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

			VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
			vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
			vertShaderStageInfo.module = vertShaderModule;
			// entrypoint
			vertShaderStageInfo.pName = "main";
			// pSpecializationInfo optionalmember to specify values for shader constants
			// default nullptr

			VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
			fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
			fragShaderStageInfo.module = fragShaderModule;
			fragShaderStageInfo.pName = "main";

			VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

			// Vertex Input
			// describes the format of the vertex data that will be passed to the vertex shader
			// bindings: spacing between data & whether data is per-vertex or per-instance
			// attribute descriptions: type of attributes passed to vertex shader, whcih binding to load 
			// 			   them from & at which offset
			VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
			vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
			// hard-coding vertex data directly into vertex shader so no vertex data
			// vertexInputInfo.vertexBindingDescriptionCount = 0;
			// vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
			// vertexInputInfo.vertexAttributeDescriptionCount = 0;
			// vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional
			auto bindingDescription = Vertex::getBindingDescription();
			auto attributeDescriptions = Vertex::getAttributeDescriptions();
			vertexInputInfo.vertexBindingDescriptionCount = 1;
			vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
			vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
			vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

			// Input assembly
			// what kind of geometry will be drawn from the vertices & if the primitive restart should be enabled
			VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
			inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
			inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
			inputAssembly.primitiveRestartEnable = VK_FALSE;
			
			/* If not dynamic
			// Viewports & scissors
			// viewport describes the region of the framebuffer that output will be rendered to
			VkViewport viewport{};
			viewport.x = 0.0f;
			viewport.y = 0.0f;
			viewport.width = (float) swapChainExtent.width;
			viewport.height = (float) swapChainExtent.height;
			viewport.minDepth = 0.0f;
			viewport.maxDepth = 1.0f;
			// scisssors define in which region pixels will actually be stored (more like filter than transform)
			VkRect2D scissor{};
			// draw entire framebuffer
			scissor.offset = {0, 0};
			scissor.extent = swapChainExtent;
			*/

			// Dynamic State
			// state to be changed without recreating pipeline at draw time
			// required to specify data at draw time
			std::vector<VkDynamicState> dynamicStates = {
			     VK_DYNAMIC_STATE_VIEWPORT,
			     VK_DYNAMIC_STATE_SCISSOR
			};
			VkPipelineDynamicStateCreateInfo dynamicState{};
			dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
			dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
			dynamicState.pDynamicStates = dynamicStates.data();

			VkPipelineViewportStateCreateInfo viewportState{};
			viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
			viewportState.viewportCount = 1;
			viewportState.scissorCount = 1;
			// if not dynamic
			// viewportState.pViewports = &viewport;
			// viewportState.pScissors = &scissor;

			// Rasterization
			VkPipelineRasterizationStateCreateInfo rasterizer{};
			rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
			rasterizer.depthClampEnable = VK_FALSE;
			rasterizer.rasterizerDiscardEnable = VK_FALSE;
			rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
			rasterizer.lineWidth = 1.0f;
			rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
			rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
			rasterizer.depthBiasEnable = VK_FALSE;
			rasterizer.depthBiasConstantFactor = 0.0f; // Optional
			rasterizer.depthBiasClamp = 0.0f; // Optional
			rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

			// Multisampling
			VkPipelineMultisampleStateCreateInfo multisampling{};
			multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
			multisampling.sampleShadingEnable = VK_FALSE;
			multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
			// Optional:
			multisampling.minSampleShading = 1.0f;
			multisampling.pSampleMask = nullptr;
			multisampling.alphaToCoverageEnable = VK_FALSE;
			multisampling.alphaToOneEnable = VK_FALSE;
			
			// Depth & stencil testing

			// Color blending
			VkPipelineColorBlendAttachmentState colorBlendAttachment{};
			colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
							    | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
			colorBlendAttachment.blendEnable = VK_FALSE;
			// optional members:
			colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
			colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
			colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
			colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
			/* pseudocode expl:
			 * if (blendEnable) {
			 *    finalColor.rgb = (srcColorBlendFactor * newColor.rgb) <colorBlendOp> (dstColorBlendFactor * oldColor.rbg);
			 *    finalColor.a = (srcAlphaBlendFactor * newColor.a) <alphaBlendOp> (dstAlphaBlendFactor * oldColor.a);
			 * } else {
			 *    finalColor = newColor;
			 * }
			 *
			 * finalColor = finalColor & colorWriteMask;
			 */

			VkPipelineColorBlendStateCreateInfo colorBlending{};
			colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
			colorBlending.logicOpEnable = VK_FALSE;
			colorBlending.logicOp = VK_LOGIC_OP_COPY; // opt
			colorBlending.attachmentCount = 1;
			colorBlending.pAttachments = &colorBlendAttachment;
			colorBlending.blendConstants[0] = 0.0f; // opt
			colorBlending.blendConstants[1] = 0.0f; // opt
			colorBlending.blendConstants[2] = 0.0f; // opt
			colorBlending.blendConstants[3] = 0.0f; // opt

			VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
			pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			// optional:
			pipelineLayoutInfo.setLayoutCount = 0;			
			pipelineLayoutInfo.pSetLayouts = nullptr;
			pipelineLayoutInfo.pushConstantRangeCount = 0;
			pipelineLayoutInfo.pPushConstantRanges = nullptr;

			if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
				throw std::runtime_error("failed to create pipeline layout.");
			}

			VkGraphicsPipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
			pipelineInfo.stageCount = 2;
			pipelineInfo.pStages = shaderStages;
			pipelineInfo.pVertexInputState = &vertexInputInfo;
			pipelineInfo.pInputAssemblyState = &inputAssembly;
			pipelineInfo.pViewportState = &viewportState;
			pipelineInfo.pRasterizationState = &rasterizer;
			pipelineInfo.pMultisampleState = &multisampling;
			pipelineInfo.pDepthStencilState = nullptr;
			pipelineInfo.pColorBlendState = &colorBlending;
			pipelineInfo.pDynamicState = &dynamicState;
			pipelineInfo.layout = pipelineLayout;
			pipelineInfo.renderPass = renderPass;
			pipelineInfo.subpass = 0;
			pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
			pipelineInfo.basePipelineIndex = -1;

			if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
				throw std::runtime_error("failed to create grphics pipeline.");
			}
	
			vkDestroyShaderModule(device, fragShaderModule, nullptr);
			vkDestroyShaderModule(device, vertShaderModule, nullptr);
		}

		VkShaderModule createShaderModule(const std::vector<char>& code) {
			VkShaderModuleCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
			createInfo.codeSize = code.size();
			createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
			
			VkShaderModule shaderModule;
			if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
				throw std::runtime_error("failed to create shader module.");
			}
			return shaderModule;
		}

		void createImageViews() {
			swapChainImageViews.resize(swapChainImages.size());

			for (size_t i = 0; i < swapChainImages.size(); i++) {
				VkImageViewCreateInfo createInfo{};
				createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
				createInfo.image = swapChainImages[i];
				createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
				createInfo.format = swapChainImageFormat;
				// default mapping
				createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
				createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
				// image's purpose & which part of image to access
				createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
				createInfo.subresourceRange.baseMipLevel = 0;
				createInfo.subresourceRange.levelCount = 1;
				createInfo.subresourceRange.baseArrayLayer = 0;
				createInfo.subresourceRange.layerCount = 1;

				if (vkCreateImageView(device, &createInfo, nullptr, &swapChainImageViews[i]) != VK_SUCCESS) {
					throw std::runtime_error("failed to create image views.");
				}
			}
		}

		void createSwapChain() {
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

			VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
			VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
			VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

			uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
			
			if (swapChainSupport.capabilities.maxImageCount > 0 && 
					imageCount > swapChainSupport.capabilities.maxImageCount) {
				imageCount = swapChainSupport.capabilities.maxImageCount;
			}

			VkSwapchainCreateInfoKHR createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
			createInfo.surface = surface;

			createInfo.minImageCount = imageCount;
			createInfo.imageFormat = surfaceFormat.format;
			createInfo.imageColorSpace = surfaceFormat.colorSpace;
			createInfo.imageExtent = extent;
			createInfo.imageArrayLayers = 1;
			createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

			QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
			uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

			if (indices.graphicsFamily != indices.presentFamily) {
				createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
				createInfo.queueFamilyIndexCount = 2;
				createInfo.pQueueFamilyIndices = queueFamilyIndices;
			} else {
				createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
				createInfo.queueFamilyIndexCount = 0; // Optional
				createInfo.pQueueFamilyIndices = nullptr; // optional
			}

			createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
			createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
			createInfo.presentMode = presentMode;
			createInfo.clipped = VK_TRUE;
			createInfo.oldSwapchain = VK_NULL_HANDLE;


			if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
				throw std::runtime_error("failed to create swap chain.");
			}

			vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
			swapChainImages.resize(imageCount);
			vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

			swapChainImageFormat = surfaceFormat.format;
			swapChainExtent = extent;
		}


		SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
			SwapChainSupportDetails details;

			// surface capabilities
			vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

			uint32_t formatCount;
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

			if (formatCount != 0) {
				details.formats.resize(formatCount);
				vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
			}

			uint32_t presentModeCount;
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

			if (presentModeCount != 0) {
				details.presentModes.resize(presentModeCount);
				vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
			}
			
			return details;
		}

		// find optimal surface format (colour depth)
		// VkSurfaceFormatKHR has format and colorSpace members
		// 	- format, e.g. VK_FORMAT_B8G8R8A8_SRGB (meaning store B, G, R, & alpha channels in that order 
		// 	with an 8 bit unsigned integer for total 32 bits / pixel
		// 	- colorSpace indicates if the SRGB color space is supported or not using the 
		// 	VK_COLOR_SPACE_SRGB_NONLINEAR_KHR flag
		// Optimal: use SRGB if it is available, as results in more accurate perceived colours, pretty much the standard.
		// 	    therefore also use SRGB color format - most common is VK_FORMAT_B8G8R8A8_SRGB
		VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
			for (const auto& availableFormat : availableFormats) {
				if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && 
				    availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
					return availableFormat;
				}
			}

			return availableFormats[0];
		}


		// find optimal presentation mode (conditions for 'swapping' images to the screen
		// Four possible presentation modes in Vulkan:
		// 	- VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR, 
		// 	VK_PRESENT_MODE_MAILBOX_KHR
		// 	- only FIFO_KHR guaranteed to be available
		VkPresentModeKHR chooseSwapPresentMode (const std::vector<VkPresentModeKHR>& availablePresentModes) {
			for (const auto& availablePresentMode : availablePresentModes) {
				// best if energy comsumption not a concern
				if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
					return availablePresentMode;
				}
			}

			// best for energy concerned devices eg mobiles
			return VK_PRESENT_MODE_FIFO_KHR;
		}

		// find optimal swap extent (resolution of images in swap chain)
		VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
			if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
				return capabilities.currentExtent;
			} else {
				int width, height;
				glfwGetFramebufferSize(window, &width, &height);

				VkExtent2D actualExtent = {
					static_cast<uint32_t>(width),
					static_cast<uint32_t>(height)
				};

				actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, 
						capabilities.maxImageExtent.width);
				actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height,
						capabilities.maxImageExtent.height);

				return actualExtent;
			}
		}

		void createSurface() {
			if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
				throw std::runtime_error("failed to create window surface.");
			}
		}

		// initialise Vulkan library by creating instance - connection between 
		// the application and the Vulkan library, and specify details of connection
		// to the driver
		void createInstance() {
			if (enableValidationLayers && !checkValidationLayerSupport()) {
				throw std::runtime_error("validation layers requested not available.");
			}
			// application information into struct, technically opt, but may provide
			// some useful info to driver to optimize our specific app e.g. because
			// it uses a well-known graphics engine with certain special behaviour
			VkApplicationInfo appInfo{};
			appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
			appInfo.pApplicationName = "Hello Triangle";
			appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
			appInfo.pEngineName = "No Engine";
			appInfo.apiVersion = VK_API_VERSION_1_0;
			// can point to extension information
			appInfo.pNext = nullptr;

			// mandatory struct to tell Vulkan driver which global extensons & 
			// validation layers to use. Global here means apply to entire program
			// & not a specific device
			VkInstanceCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
			createInfo.pApplicationInfo = &appInfo;

			auto extensions = getRequiredExtensions();
			createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
			createInfo.ppEnabledExtensionNames = extensions.data();

			VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
			// global validation layers to enable & setup debug for instance creation
			if (enableValidationLayers) {
				createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();

				populateDebugMessengerCreateInfo(debugCreateInfo);
				createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
			}
			else {
				createInfo.enabledLayerCount = 0;

				createInfo.pNext = nullptr;
			}

			// general pattern that object create function parameters in Vulkan follow:
			// 	- Pointer to struct with creation info
			// 	- Pointer to custom allocator callbacks
			// 	- Pointer to the variable that stores the handle to the new object
			// VkResult is either VK_SUCCESS or error code
			if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
				throw std::runtime_error("failed to create instance");
			}
			// if on MacOS & get VK_ERROR_INCOMPATIBLE_DRIVER, need VK_KHR_PORTABILITY_subset extension
			// see: https://vulkan-tutorial.com/en/Drawing_a_triangle/Setup/Instance#page_Encountered-VK_ERROR_INCOMPATIBLE_DRIVER
			// add:
			//
			// 	std::vector<const char*> requiredExtensions;
			//
			// 	for(uint32_t i = 0; i < glfwExtensionCount; i++) {
			//     		requiredExtensions.emplace_back(glfwExtensions[i]);
			//     	}
			//
			//     	requiredExtensions.emplace_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)
			//
			//     	createInfo.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
			//
			//     	createInfo.enabledExtensionCount = (uint32_t) requiredExtensions.size();
			//     	createInfo.ppEnabledExtensionNames = requiredExtensions.data();

		}

		void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
			createInfo = {};
			createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
			createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT
						   | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
						   | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
			createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
					       | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
					       | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
			createInfo.pfnUserCallback = debugCallback;
			createInfo.pUserData = nullptr; // Optional
		}

		void setupDebugMessenger() {
			if (!enableValidationLayers) return;
			
			VkDebugUtilsMessengerCreateInfoEXT createInfo{};
			populateDebugMessengerCreateInfo(createInfo);

			if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
				throw std::runtime_error("failed to set up debug messenger.");
			}
		}

		void pickPhysicalDevice() {
			uint32_t deviceCount = 0;
			vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

			if (deviceCount == 0) {
				throw std::runtime_error("failed to find GPUs with Vulkan support.");
			}

			std::vector<VkPhysicalDevice> devices(deviceCount);
			vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

			/** score based solution 
			 * // include <map>
			 *
			 * // use an ordered map to automatically sort by increasing score
			 * std::multimap<int, VkPhysicalDevice> candidates;
			 *
			 * for (const auto& device : devices) {
			 * int score = rateDevceSuitability(device);
			 * candidates.insert(std::make_pair(score, device));
			 * }
			 *
			 * // check if best is suitable at all
			 * if (candidates.rbegin()->first > 0) {
			 * 	physicalDevice = candidates.rbegin()->second;
			 * } else {
			 * 	throw std::runtime_error("failed to find a suitable GPU.");
			 * }
			 */
						
			for (const auto& device : devices) {
				if (isDeviceSuitable(device)) {
					physicalDevice = device;
					break;
				}
			}

			if (physicalDevice == VK_NULL_HANDLE) {
				throw std::runtime_error("failed to find a suitable GPU.");
			}
		}

		void createLogicalDevice() {
			QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
			
			std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
			std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};

			// priority required even if single queue [0.0,1.0]
			float queuePriority = 1.0f;
			for (uint32_t queueFamily : uniqueQueueFamilies) {
				// struct describing the number of queues we want for a single queue family
				VkDeviceQueueCreateInfo queueCreateInfo{};
				queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = queueFamily;
				// currently available drivers will only allow a small number of queue for each family
				// don't really need more than one as you can create all of the command buffers on
				// multiple threads and submit them all at once on the main thread in single low-overhead call
				queueCreateInfo.queueCount = 1;
				queueCreateInfo.pQueuePriorities = &queuePriority;
				queueCreateInfos.push_back(queueCreateInfo);
			}

			// struct to store set of device features to be used
			// same features that support was queried for with vkGetPhysicalDeviceFeatures
			VkPhysicalDeviceFeatures deviceFeatures{};
			// defined and left all default: VK_FALSE

			// struct for creating logical device
			VkDeviceCreateInfo createInfo{};
			createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
			createInfo.pQueueCreateInfos = queueCreateInfos.data();
			createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());;
			createInfo.pEnabledFeatures = &deviceFeatures;

			// similar to createInstance but device specific extensions & validation layers
			// device specific extension e.g. VK_KHR_swapchain which allows you to present 
			// rendered images from that device to windows (compute operation only devices won't have)
			
			createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
			createInfo.ppEnabledExtensionNames = deviceExtensions.data();
			
			// backwards compatibility as distinction between instance and device specific validation 
			// layers no longer the case in Vulkan. Up-to-date implementations will ignore 
			// enabledLayerCount & ppEnabledLayerNames
			if (enableValidationLayers) {
				createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
				createInfo.ppEnabledLayerNames = validationLayers.data();
			} else {
				createInfo.enabledLayerCount = 0;
			}

			// instantiate logical device (physical device to interface with, the queue and usage 
			// info, optional allocation callbacks pointer, pointer to varaible to store logical 
			// device handle in
			if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
				throw std::runtime_error("failed to create logical device.");
			}
			
			// get queue handle (automatically created queue with logical device)
			// if the queue famalkes are the same the handles will most likely have the same value
			vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
			vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);

		}

		bool isDeviceSuitable(VkPhysicalDevice device) {
			// basic properties (name, type, supported Vulkan version etc)
			// VkPhysicalDeviceProperties deviceProperties;
			// vkGetPhysicalDeviceProperties(device, &deviceProperties);
			// optional features (texture compression, 64 bit floats, multi viewport rendering etc)
			// VkPhysicalDeviceFeatures deviceFeatures;
			// vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
			
			// only dedicated graphics cards that support geometry shaders
			// return deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
			//	deviceFeatures.geometryShader;
			// std::cout << "Device: " << deviceProperties.deviceName << "\n"; 
			// std::cout << "Device: " << deviceProperties.deviceType << "\n"; 
			// std::cout << "Device: " << deviceProperties.vendorID << "\n"; 

			QueueFamilyIndices indices = findQueueFamilies(device);

			bool extensionsSupported = checkDeviceExtensionSupport(device);

			bool swapChainAdequate = false;
			if (extensionsSupported) {
				SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
				// at least one supported image format and presentation mode given the window surface
				swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
			}

			return indices.isComplete() && extensionsSupported && swapChainAdequate;
		}

		bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
			uint32_t extensionCount;
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

			std::vector<VkExtensionProperties> availableExtensions(extensionCount);
			vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

			std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

			for (const auto& extension : availableExtensions) {
				requiredExtensions.erase(extension.extensionName);
			}

			return requiredExtensions.empty();
		}

		QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
			// Assign index to queue families that could be found
			QueueFamilyIndices indices;

			uint32_t queueFamilyCount = 0;
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);
			std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
			vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());
			int i = 0;
			
			for (const auto& queueFamily : queueFamilies) {
				// find at least one queue family that supports VK_QUEUE_GRAPHICS_BIT
				if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
					indices.graphicsFamily = i;
				}
				// check for queue family that has capability of rpesenting to our window surface
				VkBool32 presentSupport = false;
				// (physical device, queue familiy index, surface, support result)
				vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
				if (presentSupport) {
					indices.presentFamily = i;
				}
				if (indices.isComplete()) {
					break;
				}
				i++;
			}
			return indices;
		}

		/* int rateDeviceSuitability(VkPhysicalDevice device) {
		 *	// basic properties (name, type, supported Vulkan version etc)
		 *	VkPhysicalDeviceProperties deviceProperties;
		 *	vkGetPhysicalDeviceProperties(device, &deviceProperties);
		 *	// optional features (texture compression, 64 bit floats, multi viewport rendering etc)
		 *	VkPhysicalDeviceFeatures deviceFeatures;
		 *	vkGetPhysicalDeviceFeatures(device, &deviceFeatures);
		 *	
		 * 	int score = 0;
		 *
		 *	// Discrete GPUs have a significant performance advantage
		 *	if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
		 *		score += 1000;
		 *	}
		 *
		 *	// Maximum possible size of textures affects graphics quality
		 *	score += deviceProperties.limits.maxImageDimension2D;
		 *
		 *	// Application can't function without geometry shaders
		 *	if (!deviceFeatures.geometryShader) {
		 *		return 0;
		 *	}
		 *
		 *	return score;
		 *}
		 **/


		// check instance extensions
		void checkExtensions(std::vector<const char*> reqExtensions) {
			uint32_t extensionCount = 0;
			// get number of extensions (filter by validation layer, number, details in array)
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
			// array to hold extension detaiils
			std::vector<VkExtensionProperties> extensions(extensionCount);
			// query extension details
			vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

			// list extensions
			std::cout << "available extensions:\n";
			for (const auto& extension : extensions) {
				std::cout << '\t' << extension.extensionName << '\n';
			}

			std::cout << "required extensions:\n";
			for (const char* reqExtension : reqExtensions) {
				std::cout << '\t' << reqExtension << ':';
				if (std::find_if(extensions.begin(), extensions.end(), 
						compare(reqExtension))  != extensions.end()) {
					std::cout << " AVAILABLE.\n";
				}
				else {
					std::cout << " NOT AVAILABLE.\n";
					throw std::runtime_error("extensions requested not available.");
				}
			}
		}

		struct compare
		{
			const char* key;
			compare(const char* &i): key(i) {}

			bool operator()(VkExtensionProperties &i) {
				// std::cout << i.extensionName << ' ' << key << '\n';
				if (std::strcmp(key, i.extensionName) == 0) return true;
				return false;
			}
		};

		std::vector<const char*> getRequiredExtensions() {
			uint32_t glfwExtensionCount = 0;
			const char** glfwExtensions;
			// returns extension(s) needed to interface with window system
			glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

			std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

			if (enableValidationLayers) {
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}
#ifndef NDEBUG
			// checks if required extensions are in available extensions
			checkExtensions(extensions);
#endif

			return extensions;
		}

		bool checkValidationLayerSupport() {
			uint32_t layerCount;
			vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

			std::vector<VkLayerProperties> availableLayers(layerCount);
			vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

#ifndef NDEBUG
			std::cout << "available layers:\n";
			for (const auto& layer : availableLayers) {
				std::cout << '\t' << layer.layerName << '\n';
			}
			std::cout << "requested layers:\n";
			for (const char* layerName : validationLayers) {
				std::cout << '\t' << layerName << '\n';
			}
#endif

			for (const char* layerName : validationLayers) {
				bool layerFound = false;
				for (const auto& layerProperties : availableLayers) {
					if (strcmp(layerName, layerProperties.layerName) == 0) {
						layerFound = true;
						break;
					}
				}

				if (!layerFound) {
					return false;
				}
			}
			return true;
		}

		// load binary data from file helper function
		static std::vector<char> readFile(const std::string& filename) {
			// ate: start reading at end of file, binary: read as binary file (avoid text transformation)
			std::ifstream file(filename, std::ios::ate | std::ios::binary);

			if (!file.is_open()) {
				throw std::runtime_error("failed to open file.");
			}

			// read at end means can use the read position to determine size of file & allocate a buffer
			size_t fileSize = (size_t) file.tellg();
			std::vector<char> buffer(fileSize);

			// go back to beginning & read all bytes at once
			file.seekg(0);
			file.read(buffer.data(), fileSize);

			file.close();
			return buffer;
		}

		static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
				VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
				VkDebugUtilsMessageTypeFlagsEXT messageType,
				const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
				void* pUserData) {
			std::cerr << "validation layer: " << messageType << ": " << pCallbackData->pMessage << std::endl;

			if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
				// important enough to show
			}

			return VK_FALSE;
		}

};

int main() {
	HelloTriangleApplication app;

	try {
		app.run();
	} catch (const std::exception& e) {
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
