"""Single-file interactive mirror tool for Autodesk Maya."""

from contextlib import contextmanager
from functools import wraps
import traceback
import json

import maya.cmds as cmds
import maya.mel as mel
import maya.utils as maya_utils
from maya import OpenMayaUI as omui

try:
    from PySide6 import QtCore, QtWidgets
    from shiboken6 import wrapInstance
except ImportError:
    from PySide2 import QtCore, QtWidgets
    from shiboken2 import wrapInstance


CONTROLLER_PREFIX = "mirrorController"

AXIS_DATA = {
    "X": {
        "index": 0,
        "vector": (1.0, 0.0, 0.0),
        "color": (1.0, 0.18, 0.18),
    },
    "Y": {
        "index": 1,
        "vector": (0.0, 1.0, 0.0),
        "color": (0.20, 1.0, 0.20),
    },
    "Z": {
        "index": 2,
        "vector": (0.0, 0.0, 1.0),
        "color": (0.20, 0.45, 1.0),
    },
}

_ACTIVE_JOB_ID = None
_CALLBACK_BUSY = False


@contextmanager
def _suspend_undo_recording():
    """Temporarily disable Undo recording without clearing the queue."""
    undo_was_enabled = bool(
        cmds.undoInfo(
            query=True,
            state=True,
        )
    )

    try:
        if undo_was_enabled:
            cmds.undoInfo(
                stateWithoutFlush=False,
            )

        yield

    finally:
        if undo_was_enabled:
            cmds.undoInfo(
                stateWithoutFlush=True,
            )


def _without_undo(function):
    """Run a function without adding temporary work to Maya's Undo queue."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _suspend_undo_recording():
            return function(*args, **kwargs)

    return wrapped


def _as_transform(node):
    """Return the transform represented by a DAG node."""
    if not node:
        return None

    node = node.split(".", 1)[0]

    if not cmds.objExists(node):
        return None

    if cmds.nodeType(node) in ("transform", "joint"):
        matches = cmds.ls(node, long=True) or []
        return matches[0] if matches else node

    parents = cmds.listRelatives(
        node,
        parent=True,
        fullPath=True,
    ) or []

    return parents[0] if parents else None


def _selected_transforms():
    """Return unique selected transforms in selection order."""
    selection = cmds.ls(
        selection=True,
        long=True,
        objectsOnly=True,
    ) or []

    transforms = []

    for selected in selection:
        transform = _as_transform(selected)

        if transform and transform not in transforms:
            transforms.append(transform)

    return transforms


def _mesh_shapes(transform):
    """Return non-intermediate polygon shapes."""
    return cmds.listRelatives(
        transform,
        shapes=True,
        noIntermediate=True,
        fullPath=True,
        type="mesh",
    ) or []


def _validate_target(node):
    """Validate and return a polygon target transform."""
    transform = _as_transform(node)

    if not transform or not _mesh_shapes(transform):
        raise RuntimeError(
            "The first selected object must be a polygon mesh."
        )

    return transform


def _has_parent(transform):
    """Return whether a transform has a DAG parent."""
    return bool(
        cmds.listRelatives(
            transform,
            parent=True,
            fullPath=True,
        )
    )


def _world_rotate_pivot(transform):
    """Return the exact world-space rotate pivot."""
    return tuple(
        cmds.xform(
            transform,
            query=True,
            worldSpace=True,
            rotatePivot=True,
        )
    )


def _world_rotation(transform):
    """Return the world-space Euler rotation."""
    return tuple(
        cmds.xform(
            transform,
            query=True,
            worldSpace=True,
            rotation=True,
        )
    )


def _target_center(target):
    """Return the target world bounding-box center."""
    bounds = cmds.exactWorldBoundingBox(target)

    return (
        (bounds[0] + bounds[3]) * 0.5,
        (bounds[1] + bounds[4]) * 0.5,
        (bounds[2] + bounds[5]) * 0.5,
    )


def _controller_size(target):
    """Calculate a readable controller size."""
    bounds = cmds.exactWorldBoundingBox(target)

    dimensions = (
        bounds[3] - bounds[0],
        bounds[4] - bounds[1],
        bounds[5] - bounds[2],
    )

    return max(max(dimensions) * 0.65, 1.0)


def _track_resource(controller, node):
    """Track a temporary dependency node."""
    if not cmds.attributeQuery(
        "controllerResources",
        node=controller,
        exists=True,
    ):
        cmds.addAttr(
            controller,
            longName="controllerResources",
            attributeType="message",
            multi=True,
        )

    indices = cmds.getAttr(
        controller + ".controllerResources",
        multiIndices=True,
    ) or []

    index = max(indices) + 1 if indices else 0

    cmds.connectAttr(
        node + ".message",
        "{}.controllerResources[{}]".format(
            controller,
            index,
        ),
        force=True,
    )


def _mark_controller_part(transform):
    """Mark a transform as controller geometry."""
    if not cmds.attributeQuery(
        "isMirrorControllerPart",
        node=transform,
        exists=True,
    ):
        cmds.addAttr(
            transform,
            longName="isMirrorControllerPart",
            attributeType="bool",
        )

    cmds.setAttr(
        transform + ".isMirrorControllerPart",
        True,
    )


def _create_material(
    controller,
    axis_name,
    color,
    alpha,
):
    """Create a translucent axis material."""
    shader = cmds.shadingNode(
        "surfaceShader",
        asShader=True,
        name="mirrorController_{}_MAT#".format(
            axis_name
        ),
    )

    shading_group = cmds.sets(
        renderable=True,
        noSurfaceShader=True,
        empty=True,
        name="mirrorController_{}_SG#".format(
            axis_name
        ),
    )

    cmds.connectAttr(
        shader + ".outColor",
        shading_group + ".surfaceShader",
        force=True,
    )

    cmds.setAttr(
        shader + ".outColor",
        *color,
        type="double3"
    )

    transparency = 1.0 - max(
        0.0,
        min(1.0, alpha),
    )

    cmds.setAttr(
        shader + ".outTransparency",
        transparency,
        transparency,
        transparency,
        type="double3",
    )

    _track_resource(controller, shader)
    _track_resource(controller, shading_group)

    return shading_group


def _enable_draw_on_top(transform):
    """Display controller geometry over scene geometry."""
    try:
        cmds.displaySurface(
            transform,
            xRay=True,
        )
    except RuntimeError:
        pass

    shapes = cmds.listRelatives(
        transform,
        shapes=True,
        fullPath=True,
    ) or []

    for shape in shapes:
        if cmds.attributeQuery(
            "alwaysDrawOnTop",
            node=shape,
            exists=True,
        ):
            cmds.setAttr(
                shape + ".alwaysDrawOnTop",
                True,
            )

        for attribute in (
            "castsShadows",
            "receiveShadows",
        ):
            if cmds.attributeQuery(
                attribute,
                node=shape,
                exists=True,
            ):
                cmds.setAttr(
                    "{}.{}".format(shape, attribute),
                    False,
                )


def _add_handle_metadata(
    handle,
    controller,
    axis_name,
    axis_index,
    keep_side,
):
    """Attach mirror metadata to a selectable handle."""
    cmds.addAttr(
        handle,
        longName="isMirrorHandle",
        attributeType="bool",
    )
    cmds.setAttr(
        handle + ".isMirrorHandle",
        True,
    )

    cmds.addAttr(
        handle,
        longName="mirrorAxis",
        dataType="string",
    )
    cmds.setAttr(
        handle + ".mirrorAxis",
        axis_name,
        type="string",
    )

    cmds.addAttr(
        handle,
        longName="axisIndex",
        attributeType="long",
    )
    cmds.setAttr(
        handle + ".axisIndex",
        axis_index,
    )

    cmds.addAttr(
        handle,
        longName="keepSide",
        attributeType="long",
    )
    cmds.setAttr(
        handle + ".keepSide",
        keep_side,
    )

    cmds.addAttr(
        handle,
        longName="mirrorController",
        attributeType="message",
    )

    cmds.connectAttr(
        controller + ".message",
        handle + ".mirrorController",
        force=True,
    )


def create_mirror_controller(
    target,
    position,
    rotation=(0.0, 0.0, 0.0),
    alpha=0.70,
):
    """Create a polygon mirror controller."""
    size = _controller_size(target)
    shaft_radius = max(size * 0.018, 0.025)
    handle_radius = max(size * 0.09, 0.12)

    controller = cmds.group(
        empty=True,
        world=True,
        name=CONTROLLER_PREFIX + "#",
    )

    cmds.addAttr(
        controller,
        longName="isMirrorController",
        attributeType="bool",
    )
    cmds.setAttr(
        controller + ".isMirrorController",
        True,
    )

    for axis_name, data in AXIS_DATA.items():
        shading_group = _create_material(
            controller,
            axis_name,
            data["color"],
            alpha,
        )

        shaft = cmds.polyCylinder(
            radius=shaft_radius,
            height=size * 2.0,
            subdivisionsAxis=12,
            subdivisionsHeight=1,
            axis=data["vector"],
            constructionHistory=False,
            name="mirrorController_{}_shaft#".format(
                axis_name
            ),
        )[0]

        cmds.parent(
            shaft,
            controller,
            relative=True,
        )

        _mark_controller_part(shaft)

        cmds.sets(
            shaft,
            edit=True,
            forceElement=shading_group,
        )

        _enable_draw_on_top(shaft)

        shaft_shapes = cmds.listRelatives(
            shaft,
            shapes=True,
            fullPath=True,
        ) or []

        for shape in shaft_shapes:
            cmds.setAttr(
                shape + ".overrideEnabled",
                True,
            )
            cmds.setAttr(
                shape + ".overrideDisplayType",
                2,
            )

        for keep_side, side_name in (
            (1, "positive"),
            (-1, "negative"),
        ):
            handle = cmds.polySphere(
                radius=handle_radius,
                subdivisionsAxis=16,
                subdivisionsHeight=10,
                constructionHistory=False,
                name="mirrorController_{}_{}#".format(
                    axis_name,
                    side_name,
                ),
            )[0]

            cmds.parent(
                handle,
                controller,
                relative=True,
            )

            _mark_controller_part(handle)

            vector = data["vector"]

            cmds.setAttr(
                handle + ".translate",
                vector[0] * size * keep_side,
                vector[1] * size * keep_side,
                vector[2] * size * keep_side,
                type="double3",
            )

            cmds.sets(
                handle,
                edit=True,
                forceElement=shading_group,
            )

            _enable_draw_on_top(handle)

            _add_handle_metadata(
                handle,
                controller,
                axis_name,
                data["index"],
                keep_side,
            )

    # The controller is matched only during creation.
    # No target parenting or coordinate reset happens here.
    cmds.xform(
        controller,
        worldSpace=True,
        rotation=rotation,
    )

    cmds.xform(
        controller,
        worldSpace=True,
        translation=position,
    )

    return controller


def _kill_job(job_id):
    """Kill a scriptJob after its callback has finished."""
    global _ACTIVE_JOB_ID

    if job_id and cmds.scriptJob(exists=job_id):
        cmds.scriptJob(
            kill=job_id,
            force=True,
        )

    if _ACTIVE_JOB_ID == job_id:
        _ACTIVE_JOB_ID = None


def _unparent_foreign_children(controller):
    """Protect non-controller children before controller deletion."""
    children = cmds.listRelatives(
        controller,
        children=True,
        type="transform",
        fullPath=True,
    ) or []

    for child in children:
        is_controller_part = cmds.attributeQuery(
            "isMirrorControllerPart",
            node=child,
            exists=True,
        )

        if not is_controller_part:
            try:
                cmds.parent(
                    child,
                    world=True,
                    absolute=True,
                )
            except RuntimeError:
                pass


@_without_undo
def _delete_controller(
    controller,
    kill_job=True,
):
    """Delete a controller and restore temporary scene changes."""
    if not controller or not cmds.objExists(controller):
        return

    job_id = None

    if cmds.attributeQuery(
        "selectionJob",
        node=controller,
        exists=True,
    ):
        job_id = cmds.getAttr(
            controller + ".selectionJob"
        )

    if kill_job:
        _kill_job(job_id)

    # Restore the target before deleting controller metadata.
    _restore_target_selection(controller)

    resources = []

    if cmds.attributeQuery(
        "controllerResources",
        node=controller,
        exists=True,
    ):
        resources = cmds.listConnections(
            controller + ".controllerResources",
            source=True,
            destination=False,
        ) or []

    _unparent_foreign_children(controller)

    cmds.delete(controller)

    resources = [
        node for node in set(resources)
        if cmds.objExists(node)
    ]

    if resources:
        cmds.delete(resources)


def cancel_mirror_session():
    """Remove active controllers and callbacks."""
    global _ACTIVE_JOB_ID
    global _CALLBACK_BUSY

    _kill_job(_ACTIVE_JOB_ID)

    _ACTIVE_JOB_ID = None
    _CALLBACK_BUSY = False

    candidates = cmds.ls(
        CONTROLLER_PREFIX + "*",
        type="transform",
        long=True,
    ) or []

    for candidate in candidates:
        if not cmds.objExists(candidate):
            continue

        if cmds.attributeQuery(
            "isMirrorController",
            node=candidate,
            exists=True,
        ):
            _delete_controller(candidate)


def _selected_handle(controller):
    """Return metadata from the selected handle."""
    selection = cmds.ls(
        selection=True,
        long=True,
    ) or []

    for selected in selection:
        handle = _as_transform(selected)

        if not handle or not cmds.objExists(handle):
            continue

        if not cmds.attributeQuery(
            "isMirrorHandle",
            node=handle,
            exists=True,
        ):
            continue

        connected = cmds.listConnections(
            handle + ".mirrorController",
            source=True,
            destination=False,
            type="transform",
        ) or []

        if not connected:
            continue

        connected_long = (
            cmds.ls(connected[0], long=True)
            or connected
        )

        controller_long = (
            cmds.ls(controller, long=True)
            or [controller]
        )

        if connected_long[0] != controller_long[0]:
            continue

        return {
            "axis": cmds.getAttr(
                handle + ".mirrorAxis"
            ),
            "axisIndex": int(
                cmds.getAttr(handle + ".axisIndex")
            ),
            "keepSide": int(
                cmds.getAttr(handle + ".keepSide")
            ),
        }

    return None


def _find_new_mirror_node(
    existing_nodes,
    target,
):
    """Find the polyMirror node created by Maya."""
    current_nodes = set(
        cmds.ls(type="polyMirror") or []
    )

    new_nodes = list(
        current_nodes.difference(existing_nodes)
    )

    if not new_nodes:
        return None

    history = set(
        cmds.listHistory(
            target,
            pruneDagObjects=True,
        ) or []
    )

    matches = [
        node for node in new_nodes
        if node in history
    ]

    return matches[-1] if matches else new_nodes[-1]


def _enum_index(
    node,
    attribute,
    label,
    fallback,
):
    """Find an enum index by label."""
    enum_data = cmds.attributeQuery(
        attribute,
        node=node,
        listEnum=True,
    ) or []

    if enum_data:
        labels = enum_data[0].split(":")
        requested = label.lower().replace(" ", "")

        for index, enum_label in enumerate(labels):
            normalized = (
                enum_label.lower().replace(" ", "")
            )

            if requested in normalized:
                return index

    return fallback


def _configure_world_mirror(
    mirror_node,
    axis_index,
    keep_side,
):
    """Configure a World Mirror at the world origin."""
    world_mode = _enum_index(
        mirror_node,
        "mirrorAxis",
        "world",
        2,
    )

    # Positive handle keeps the positive side.
    axis_direction = (
        1 if keep_side > 0 else 0
    )

    cmds.polyMirrorFace(
        mirror_node,
        edit=True,
        worldSpace=True,
        mirrorAxis=world_mode,
        axis=int(axis_index),
        axisDirection=axis_direction,
        pivot=(0.0, 0.0, 0.0),
    )

    cmds.dgdirty(mirror_node)


def _parent_target_to_controller(
    target,
    controller,
):
    """Parent the target while preserving its world transform."""
    result = cmds.parent(
        target,
        controller,
        absolute=True,
    ) or []

    if not result:
        raise RuntimeError(
            "Unable to parent the target to the controller."
        )

    return result[0]


def _unparent_target(target):
    """Unparent the target while preserving its world transform."""
    if not target or not cmds.objExists(target):
        return target

    result = cmds.parent(
        target,
        world=True,
        absolute=True,
    ) or []

    if not result:
        raise RuntimeError(
            "Unable to unparent the mirrored target."
        )

    return result[0]


def _move_controller_to_world(controller):
    """Align the controller coordinate frame to World."""
    cmds.xform(
        controller,
        worldSpace=True,
        rotation=(0.0, 0.0, 0.0),
    )

    cmds.xform(
        controller,
        worldSpace=True,
        translation=(0.0, 0.0, 0.0),
    )


def _restore_controller(
    controller,
    position,
    rotation,
):
    """Restore the original controller coordinate frame."""
    cmds.xform(
        controller,
        worldSpace=True,
        rotation=rotation,
    )

    cmds.xform(
        controller,
        worldSpace=True,
        translation=position,
    )

def _protect_target_selection(
    controller,
    target,
):
    """Make the target visible but temporarily unselectable."""
    nodes = [target]

    nodes.extend(
        cmds.listRelatives(
            target,
            shapes=True,
            noIntermediate=True,
            fullPath=True,
        ) or []
    )

    display_state = []

    for node in nodes:
        if not cmds.objExists(node):
            continue

        if not cmds.attributeQuery(
            "overrideEnabled",
            node=node,
            exists=True,
        ):
            continue

        if not cmds.attributeQuery(
            "overrideDisplayType",
            node=node,
            exists=True,
        ):
            continue

        display_state.append(
            {
                "node": node,
                "overrideEnabled": bool(
                    cmds.getAttr(
                        node + ".overrideEnabled"
                    )
                ),
                "overrideDisplayType": int(
                    cmds.getAttr(
                        node + ".overrideDisplayType"
                    )
                ),
            }
        )

        cmds.setAttr(
            node + ".overrideEnabled",
            True,
        )

        # Reference mode keeps the object visible but unselectable.
        cmds.setAttr(
            node + ".overrideDisplayType",
            2,
        )

    if not cmds.attributeQuery(
        "targetDisplayState",
        node=controller,
        exists=True,
    ):
        cmds.addAttr(
            controller,
            longName="targetDisplayState",
            dataType="string",
        )

    cmds.setAttr(
        controller + ".targetDisplayState",
        json.dumps(display_state),
        type="string",
    )


def _restore_target_selection(controller):
    """Restore the target's original display overrides."""
    if not controller or not cmds.objExists(controller):
        return

    if not cmds.attributeQuery(
        "targetDisplayState",
        node=controller,
        exists=True,
    ):
        return

    serialized_state = cmds.getAttr(
        controller + ".targetDisplayState"
    )

    if not serialized_state:
        return

    try:
        display_state = json.loads(
            serialized_state
        )
    except (TypeError, ValueError):
        return

    for item in display_state:
        node = item.get("node")

        if not node or not cmds.objExists(node):
            continue

        if cmds.attributeQuery(
            "overrideEnabled",
            node=node,
            exists=True,
        ):
            cmds.setAttr(
                node + ".overrideEnabled",
                item.get(
                    "overrideEnabled",
                    False,
                ),
            )

        if cmds.attributeQuery(
            "overrideDisplayType",
            node=node,
            exists=True,
        ):
            cmds.setAttr(
                node + ".overrideDisplayType",
                item.get(
                    "overrideDisplayType",
                    0,
                ),
            )

    cmds.setAttr(
        controller + ".targetDisplayState",
        "",
        type="string",
    )


def _execute_mirror_deferred(
    target,
    controller,
    job_id,
    mode,
    handle_data,
    controller_position,
    controller_rotation,
):
    """Execute Mirror without restoring the visible controller on undo."""
    global _CALLBACK_BUSY

    history_enabled = bool(
        cmds.constructionHistory(
            query=True,
            toggle=True,
        )
    )

    undo_open = False
    coordinate_group = None
    working_target = target
    target_is_parented = False
    group_is_world_aligned = False

    try:
        # The callback has finished, so its scriptJob can now be killed.
        _kill_job(job_id)

        if not cmds.objExists(target):
            raise RuntimeError(
                "The mirror target no longer exists."
            )

        if not cmds.objExists(controller):
            raise RuntimeError(
                "The mirror controller no longer exists."
            )

        # Delete the visible controller before opening the Mirror chunk.
        # A single undo will therefore not restore the controller.
        _delete_controller(
            controller,
            kill_job=False,
        )

        controller = None

        if not history_enabled:
            with _suspend_undo_recording():
                cmds.constructionHistory(toggle=True)

        cmds.undoInfo(
            openChunk=True,
            chunkName="Controller Native Mirror",
        )
        undo_open = True

        if mode == "custom":
            # Create an invisible replacement for the controller frame.
            coordinate_group = _create_coordinate_group(
                controller_position,
                controller_rotation,
            )

            working_target = _parent_target_to_controller(
                target,
                coordinate_group,
            )

            target_is_parented = True

            # Align the custom coordinate frame to World.
            _move_controller_to_world(
                coordinate_group
            )

            group_is_world_aligned = True

            cmds.dgdirty(allPlugs=True)
            cmds.refresh(force=True)

        existing_nodes = set(
            cmds.ls(type="polyMirror") or []
        )

        cmds.select(
            working_target,
            replace=True,
        )

        mel.eval("performPolyMirror 0;")

        mirror_node = _find_new_mirror_node(
            existing_nodes,
            working_target,
        )

        if not mirror_node:
            raise RuntimeError(
                "Maya did not create a polyMirror node."
            )

        _configure_world_mirror(
            mirror_node,
            handle_data["axisIndex"],
            handle_data["keepSide"],
        )

        cmds.dgdirty(allPlugs=True)
        cmds.refresh(force=True)

        if mode == "custom":
            # Restore the temporary coordinate frame.
            _restore_controller(
                coordinate_group,
                controller_position,
                controller_rotation,
            )

            group_is_world_aligned = False

            cmds.dgdirty(allPlugs=True)
            cmds.refresh(force=True)

            working_target = _unparent_target(
                working_target
            )

            target_is_parented = False

            if cmds.objExists(coordinate_group):
                cmds.delete(coordinate_group)

            coordinate_group = None

        cmds.select(
            working_target,
            replace=True,
        )

        cmds.inViewMessage(
            assistMessage=(
                "Mirror completed on {} axis.".format(
                    handle_data["axis"]
                )
            ),
            position="midCenterTop",
            fade=True,
        )

    except Exception as error:
        cmds.warning(
            "Mirror failed: {}".format(error)
        )
        traceback.print_exc()

    finally:
        # Restore the target if execution stopped midway.
        if (
            coordinate_group
            and cmds.objExists(coordinate_group)
        ):
            try:
                if group_is_world_aligned:
                    _restore_controller(
                        coordinate_group,
                        controller_position,
                        controller_rotation,
                    )

                    cmds.dgdirty(allPlugs=True)
                    cmds.refresh(force=True)

                if (
                    target_is_parented
                    and working_target
                    and cmds.objExists(working_target)
                ):
                    working_target = _unparent_target(
                        working_target
                    )

                if cmds.objExists(coordinate_group):
                    cmds.delete(coordinate_group)

            except Exception:
                traceback.print_exc()

        if controller and cmds.objExists(controller):
            try:
                _delete_controller(
                    controller,
                    kill_job=False,
                )
            except Exception:
                traceback.print_exc()

        if undo_open:
            cmds.undoInfo(closeChunk=True)

        if not history_enabled:
            with _suspend_undo_recording():
                cmds.constructionHistory(toggle=False)

        _CALLBACK_BUSY = False


def _on_handle_selected(
    target,
    controller,
    job_id,
    mode,
    controller_position,
    controller_rotation,
):
    """Record the handle direction and defer all scene changes."""
    global _CALLBACK_BUSY

    if _CALLBACK_BUSY:
        return

    if not cmds.objExists(controller):
        return

    handle_data = _selected_handle(controller)

    if not handle_data:
        return

    # Prevent repeated callbacks before deferred execution.
    _CALLBACK_BUSY = True

    maya_utils.executeDeferred(
        lambda: _execute_mirror_deferred(
            target,
            controller,
            job_id,
            mode,
            handle_data,
            controller_position,
            controller_rotation,
        )
    )

def _create_coordinate_group(
    position,
    rotation,
):
    """Create an invisible coordinate group matching the controller."""
    group = cmds.group(
        empty=True,
        world=True,
        name="mirrorCoordinateSpace#",
    )

    cmds.xform(
        group,
        worldSpace=True,
        rotation=rotation,
    )

    cmds.xform(
        group,
        worldSpace=True,
        translation=position,
    )

    return group


@_without_undo
def start_mirror_session(
    target=None,
    mode="world",
):
    """Create a controller and wait for a handle selection."""
    global _ACTIVE_JOB_ID
    global _CALLBACK_BUSY

    mode = str(mode).strip().lower()

    if mode not in ("world", "custom"):
        raise RuntimeError(
            "Mirror mode must be World or Custom."
        )

    transforms = _selected_transforms()

    if target is not None:
        target_transform = _validate_target(target)

    elif transforms:
        target_transform = _validate_target(
            transforms[0]
        )

    else:
        raise RuntimeError(
            "Select a polygon mesh to mirror."
        )

    if _has_parent(target_transform):
        raise RuntimeError(
            "The current version requires an unparented polygon."
        )

    if mode == "custom":
        if len(transforms) not in (1, 2):
            raise RuntimeError(
                "Custom mode requires one or two objects. "
                "Select the polygon target first and, optionally, "
                "a reference object second."
            )

        # With one selected object, use the target itself as the
        # coordinate reference. This is equivalent to using a duplicate
        # of the target as the second selected reference object.
        reference = (
            transforms[1]
            if len(transforms) == 2
            else target_transform
        )

        if (
            len(transforms) == 2
            and reference == target_transform
        ):
            raise RuntimeError(
                "The target and reference must be different objects."
            )

        # Only record the reference world position and rotation here.
        controller_position = _world_rotate_pivot(
            reference
        )

        controller_rotation = _world_rotation(
            reference
        )

    else:
        if len(transforms) > 1:
            raise RuntimeError(
                "World mode requires one selected polygon."
            )

        # World mode always displays its controller at the world origin.
        controller_position = (
            0.0,
            0.0,
            0.0,
        )

        controller_rotation = (
            0.0,
            0.0,
            0.0,
        )

    cancel_mirror_session()
    _CALLBACK_BUSY = False

    # Only create and match the controller at this stage.
    controller = create_mirror_controller(
        target=target_transform,
        position=controller_position,
        rotation=controller_rotation,
    )

    # Prevent the target mesh from blocking controller selection.
    _protect_target_selection(
        controller,
        target_transform,
    )



    cmds.select(clear=True)

    job_holder = {
        "id": None,
    }

    callback = lambda: _on_handle_selected(
        target_transform,
        controller,
        job_holder["id"],
        mode,
        controller_position,
        controller_rotation,
    )

    job_holder["id"] = cmds.scriptJob(
        event=[
            "SelectionChanged",
            callback,
        ],
        protected=True,
    )

    _ACTIVE_JOB_ID = job_holder["id"]

    cmds.addAttr(
        controller,
        longName="selectionJob",
        attributeType="long",
    )

    cmds.setAttr(
        controller + ".selectionJob",
        _ACTIVE_JOB_ID,
    )

    cmds.inViewMessage(
        assistMessage=(
            "{} Mirror: choose the side to keep.".format(
                mode.title()
            )
        ),
        position="midCenterTop",
        fade=True,
    )

    return controller


WINDOW_OBJECT_NAME = "mirrorToolWindow"
_WINDOW_INSTANCE = None


def get_maya_main_window():
    """Return Maya's main window as a Qt widget."""
    pointer = omui.MQtUtil.mainWindow()

    if pointer is None:
        return None

    return wrapInstance(
        int(pointer),
        QtWidgets.QWidget,
    )


class MirrorToolWindow(QtWidgets.QDialog):
    """Display the Mirror Tool interface."""

    def __init__(self, parent=None):
        if parent is None:
            parent = get_maya_main_window()

        super().__init__(parent)

        self.setObjectName(WINDOW_OBJECT_NAME)
        self.setWindowTitle("Mirror Tool")
        self.setFixedSize(290, 145)

        self._build_ui()
        self._create_connections()

    def _build_ui(self):
        """Build the user interface."""
        self.setStyleSheet(
            """
            QDialog {
                background-color: #25282d;
            }

            QLabel {
                color: #c9cdd2;
                font-size: 12px;
            }

            QComboBox {
                color: #f2f2f2;
                background-color: #34383e;
                border: 1px solid #4b5057;
                border-radius: 5px;
                padding: 6px 10px;
                min-height: 22px;
            }

            QComboBox:hover {
                border-color: #6b9bc6;
            }

            QComboBox::drop-down {
                border: none;
                width: 24px;
            }

            QComboBox QAbstractItemView {
                color: #f2f2f2;
                background-color: #34383e;
                selection-background-color: #3d78b8;
                outline: none;
            }

            QPushButton {
                color: #ffffff;
                background-color: #3d78b8;
                border: 1px solid #5594d6;
                border-radius: 7px;
                padding: 9px;
                font-size: 14px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: #4a8bd0;
                border-color: #70aeeb;
            }

            QPushButton:pressed {
                background-color: #2f6499;
            }
            """
        )

        mode_label = QtWidgets.QLabel(
            "Coordinate System"
        )

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(
            ["World", "Custom"]
        )

        self.mirror_button = QtWidgets.QPushButton(
            "Mirror"
        )
        self.mirror_button.setMinimumHeight(42)

        if hasattr(QtCore.Qt, "CursorShape"):
            cursor = (
                QtCore.Qt.CursorShape.PointingHandCursor
            )
        else:
            cursor = QtCore.Qt.PointingHandCursor

        self.mirror_button.setCursor(cursor)

        mode_layout = QtWidgets.QHBoxLayout()
        mode_layout.setSpacing(12)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo, 1)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(
            18,
            16,
            18,
            18,
        )
        main_layout.setSpacing(14)
        main_layout.addLayout(mode_layout)
        main_layout.addWidget(self.mirror_button)

    def _create_connections(self):
        """Connect interface signals."""
        self.mirror_button.clicked.connect(
            self._start_mirror
        )

    def _start_mirror(self):
        """Start an interactive Mirror session."""
        try:
            start_mirror_session(
                mode=self.mode_combo.currentText().lower()
            )

        except Exception as error:
            QtWidgets.QMessageBox.warning(
                self,
                "Mirror Tool",
                str(error),
            )


def show():
    """Show the Mirror Tool window."""
    global _WINDOW_INSTANCE

    if _WINDOW_INSTANCE is not None:
        try:
            _WINDOW_INSTANCE.close()
            _WINDOW_INSTANCE.deleteLater()
        except RuntimeError:
            pass

    maya_window = get_maya_main_window()

    if maya_window is not None:
        existing = maya_window.findChild(
            QtWidgets.QDialog,
            WINDOW_OBJECT_NAME,
        )

        if existing is not None:
            existing.close()
            existing.deleteLater()

    _WINDOW_INSTANCE = MirrorToolWindow(
        parent=maya_window
    )

    _WINDOW_INSTANCE.show()
    _WINDOW_INSTANCE.raise_()
    _WINDOW_INSTANCE.activateWindow()

    return _WINDOW_INSTANCE


if __name__ == "__main__":
    show()
