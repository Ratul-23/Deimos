from typing import Union, Optional

from wizwalker.memory.memory_object import Primitive, MemoryObject
from .camera_controller import (
    DynamicCameraController,
    CameraController,
    DynamicFreeCameraController,
    DynamicElasticCameraController,
)
from .client_object import DynamicClientObject
from .character_registry import DynamicCharacterRegistry
from .enums import AccountPermissions
from .gamebryo_presenter import DynamicGamebryoPresenter
from .fishing_manager import FishingManager


# note: not defined
class GameClient(MemoryObject):
    async def read_base_address(self) -> int:
        raise NotImplementedError()

    async def elastic_camera_controller(self) -> Optional[DynamicElasticCameraController]:
        offset = await self.pattern_scan_offset_cached(
            rb"\x48\x8B\x8F....\xF3\x0F\x10.....\xF3\x0F\x10.....\x48\x85\xC9\x74.",
            3,
            "elastic_camera_controller",
            0x222C8
        )

        addr = await self.read_value_from_offset(offset, Primitive.uint64)

        if addr == 0:
            return None

        return DynamicElasticCameraController(self.hook_handler, addr)

    async def free_camera_controller(self) -> Optional[DynamicFreeCameraController]:
        offset = await self.pattern_scan_offset_cached(
            rb"\x48\x8B\x8F....\x48\x8B\x01\x33\xD2\xFF\x50.\xB9",
            3,
            "free_camera_controller",
            0x222D8
        )

        addr = await self.read_value_from_offset(offset, Primitive.uint64)

        if addr == 0:
            return None

        return DynamicFreeCameraController(self.hook_handler, addr)

    async def selected_camera_controller(self) -> Optional[DynamicCameraController]:
        """
        The in use camera controller
        """
        offset = await self.pattern_scan_offset_cached(
            rb"\xC7\x41.....\x48\x8B\xBE....\x48\x85\xFF\x0F\x84",
            10,
            "selected_camera_controller",
            0x222F8
        )

        addr = await self.read_value_from_offset(offset, Primitive.uint64)

        if addr == 0:
            return None

        return DynamicCameraController(self.hook_handler, addr)

    async def write_selected_camera_controller(self, selected_camera_controller: Union[CameraController, int]):
        """
        Write the in use camera controller
        """
        if isinstance(selected_camera_controller, CameraController):
            selected_camera_controller = await selected_camera_controller.read_base_address()

        offset = await self.pattern_scan_offset_cached(
            rb"\xC7\x41.....\x48\x8B\xBE....\x48\x85\xFF\x0F\x84",
            10,
            "selected_camera_controller",
            0x222F8
        )

        await self.write_value_to_offset(offset, selected_camera_controller, Primitive.uint64)

    async def is_freecam(self) -> bool:
        """
        If the game is currently in freecam mode
        """
        offset = await self.pattern_scan_offset_cached(
            rb"\x0F\xB6\x88....\x88\x8B....\x84\xC9\x0F\x85....\x48\x8D\x55.\x48\x8B\xCB\xE8",
            3,
            "is_freecam",
            0x22310
        )
        return await self.read_value_from_offset(offset, Primitive.bool)

    async def write_is_freecam(self, is_freecam: bool):
        """
        Write if the game is currently in freecam mode
        """
        offset = await self.pattern_scan_offset_cached(
            rb"\x0F\xB6\x88....\x88\x8B....\x84\xC9\x0F\x85....\x48\x8D\x55.\x48\x8B\xCB\xE8",
            3,
            "is_freecam",
            0x22310
        )
        await self.write_value_to_offset(offset, is_freecam, Primitive.bool)

    async def root_client_object(self) -> Optional[DynamicClientObject]:
        """
        The root client object, all other client objects are its children
        """
        offset = await self.pattern_scan_offset_cached(
            rb"\x48\x8D\x93....\xFF\x90....\x90\x48\x8B\x7C\x24.\x48\x85\xFF\x74.\xBE\xFF\xFF\xFF\xFF\x8B\xC6\xF0\x0F\xC1\x47\x08",
            3,
            "root_client_object",
            0x21300
        )

        addr = await self.read_value_from_offset(offset, Primitive.uint64)

        if addr == 0:
            return None

        return DynamicClientObject(self.hook_handler, addr)

    async def frames_per_second(self) -> float:
        """
        The number of frames processed the last second, updated every 5 seconds by default
        """
        offset = await self.pattern_scan_offset_cached(
            rb"\xF3\x0F\x11\x8B....\xC7\x05........\xF2\x0F\x11.....\x48\x8B\x8B....\x48\x85\xC9\x74.",
            4,
            "frames_per_second",
            0x21A7C
        )
        return await self.read_value_from_offset(offset, Primitive.float32)

    async def shutdown_signal(self) -> int:
        """
        Signal used to check if the main loop should close
        """
        offset = await self.pattern_scan_offset_cached(
            rb"\x38\x9F....\x74.\xE8....\x83\xF8\x64\x0F\x8F....\xB9\x0F\x00\x00\x00",
            2,
            "shutdown_signal",
            0x211B8
        )
        return await self.read_value_from_offset(offset, Primitive.int32)

    async def write_shutdown_signal(self, shutdown_signal: int):
        """
        Writing 1 into the shutdown signal will close the program (exits main loop)
        """
        offset = await self.pattern_scan_offset_cached(
            rb"\x38\x9F....\x74.\xE8....\x83\xF8\x64\x0F\x8F....\xB9\x0F\x00\x00\x00",
            2,
            "shutdown_signal",
            0x211B8
        )
        await self.write_value_to_offset(offset, shutdown_signal, Primitive.int32)

    async def user_id(self) -> int:
        """Account-level UserID from MSG_USER_VALIDATE / MSG_CHARACTERSELECTED."""
        return await self.read_value_from_offset(0x21258, Primitive.uint64)

    async def player_gid(self) -> int:
        """Current character's global ID (CharID from MSG_CHARACTERSELECTED)."""
        return await self.read_value_from_offset(0x214C0, Primitive.uint64)

    async def machine_id(self) -> int:
        """MachineID sent during login."""
        return await self.read_value_from_offset(0x214C8, Primitive.uint64)

    async def character_registry(self) -> Optional[DynamicCharacterRegistry]:
        """
        Get the character registry
        """
        # TODO: find where this loaded in for offset pattern
        addr = await self.read_value_from_offset(0x224A8, Primitive.uint64)

        if addr == 0:
            return None

        return DynamicCharacterRegistry(self.hook_handler, addr)

    async def account_permissions(self) -> AccountPermissions:
        offset = await self.pattern_scan_offset_cached(
            rb"\x41\x89\x86....\x4D\x8B\x06\x8B\xD0\x49\x8B\xCE\x41\xFF\x90....\x49\x8B\x06\x49\x8B\xCE\xFF\x90....",
            3,
            "account_permissions",
            0x21DC4
        )
        return await self.read_enum(offset, AccountPermissions)

    async def write_account_permissions(self, account_permissions: AccountPermissions):
        offset = await self.pattern_scan_offset_cached(
            rb"\x41\x89\x86....\x4D\x8B\x06\x8B\xD0\x49\x8B\xCE\x41\xFF\x90....\x49\x8B\x06\x49\x8B\xCE\xFF\x90....",
            3,
            "account_permissions",
            0x21DC4
        )
        await self.write_enum(offset, account_permissions)

    async def has_membership(self) -> bool:
        offset = await self.pattern_scan_offset_cached(
            rb"\x83\xBB....\x00\x75\x04\xB2\x01\xEB\x02\x33\xD2\x48\x8B\x0D....\xE8",
            2,
            "has_membership",
            0x21DC8
        )
        return await self.read_value_from_offset(offset, Primitive.bool)

    # no, this doesn't let you go in membership areas
    async def write_has_membership(self, has_membership: bool):
        offset = await self.pattern_scan_offset_cached(
            rb"\x83\xBB....\x00\x75\x04\xB2\x01\xEB\x02\x33\xD2\x48\x8B\x0D....\xE8",
            2,
            "has_membership",
            0x21DC8
        )
        await self.write_value_to_offset(offset, has_membership, Primitive.bool)

    async def gamebryo_presenter(self) -> DynamicGamebryoPresenter:
        """
        Thing used for rendering
        """
        offset = await self.pattern_scan_offset_cached(
            rb"\x48\x89\x8B....\x48\x8B\x01\xFF\x50.\x84\xC0\x75.\xE8",
            3,
            "gamebryo_presenter",
            0x222A8
        )

        addr = await self.read_value_from_offset(offset, Primitive.uint64)

        if addr == 0:
            return None

        return DynamicGamebryoPresenter(self.hook_handler, addr)

    async def fishing_manager(self) -> FishingManager:
        addr = await self.read_value_from_offset(0x231a8, Primitive.uint64)
        return FishingManager(self.hook_handler, addr)

class CurrentGameClient(GameClient):
    _base_address = None

    async def read_base_address(self) -> int:
        if self._base_address is not None:
            return self._base_address

        addr = await self.pattern_scan(rb"\x48\x8b.....\x48\x8b.\x80\xb8....\x00\x74.\x4c\x8b", module="WizardGraphicalClient.exe")
        offset = await self.read_typed(addr + 3, Primitive.int32)

        self._base_address = await self.read_typed(addr + 7 + offset, Primitive.uint64)
        return self._base_address
